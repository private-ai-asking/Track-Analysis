from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer
from track_analysis.components.track_analysis.features.scrobbling.algorithm.cache_builder_pipeline import CacheBuilderPipeline
from track_analysis.components.track_analysis.features.scrobbling.utils.cache_helper import ScrobbleCacheHelper
from track_analysis.components.track_analysis.features.scrobbling.embedding.embedding_searcher import EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext
from track_analysis.components.track_analysis.features.scrobbling.model.scrabble_cache_algorithm_parameters import ScrobbleCacheAlgorithmParameters
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_utility import ScrobbleUtility


class ParameterTester:
    """
    Grid-search parameters with HARD constraints:
      - FA == 0 (no false-accepts)
      - FR == 0 (no false-rejects)
    Then maximize coverage = (TA + TR) / total_gold.
    Uses combined FAISS index and reduced parameter grid.
    """

    def __init__(
            self,
            logger: HoornLogger,
            gold_standard_path: Path,
            data_loader: ScrobbleDataLoader,
            scrobble_utils: ScrobbleUtility,
            cache_helper: ScrobbleCacheHelper,
            embedding_model: SentenceTransformer,
            manual_override_path: Path,
            searcher: EmbeddingSearcher,
            embed_weights: Dict,
            scorer: SimilarityScorer,
            test_mode: bool = False
    ):
        self._logger = logger
        self._separator = "ParameterTester"
        self._logger.trace("Initialized ParameterTester.", separator=self._separator)

        # Gold standard
        self._gold_df = pd.read_csv(gold_standard_path)
        self._gold_standard_path = gold_standard_path
        self._total_gold = len(self._gold_df)

        # Data loader & lookup
        self._data_loader = data_loader
        self._library_index = data_loader.get_index()
        self._library_keys = data_loader.get_keys()
        self._library_lookup = None  # filled after load
        self._library_df = None

        self._scrobble_utils = scrobble_utils
        self._scorer = scorer
        self._searcher = searcher
        self._embedder = embedding_model
        self._test_mode = test_mode

        # Reduced hyperparameter grids
        self._confidence_accept_grid = [60, 65, 70, 75, 80]
        self._confidence_reject_grid = [20, 30, 40]
        self._sigma_grid = [0.1, 0.2, 0.3]
        self._token_grid = [60, 65, 70]
        self._embeds_grid = [
            {'title': 0.3, 'album': 0.4, 'artist': 0.3 },
            {'title': 0.35, 'album': 0.4, 'artist': 0.25 }
        ]

        self._cache_helper = cache_helper
        self._override_path = manual_override_path

        self._weights = embed_weights

    def tune(self) -> None:
        # Load data once
        self._library_lookup = self._data_loader.get_direct_lookup()
        self._library_df = self._data_loader.get_library_data()

        results: List[Dict[str, Any]] = []

        # iterate reduced grid
        for ca in self._confidence_accept_grid:
            for cr in self._confidence_reject_grid:
                for sigma in self._sigma_grid:
                    for tok in self._token_grid:
                        for embed_weight in self._embeds_grid:
                            params = ScrobbleCacheAlgorithmParameters(
                                confidence_accept_threshold=ca,
                                confidence_reject_threshold=cr,
                                gaussian_sigma=sigma,
                                gold_standard_csv_path=self._gold_standard_path,
                                embed_weights=embed_weight,
                                token_accept_threshold=tok,
                                manual_override_path=self._override_path
                            )

                            # build pipeline
                            pipeline = CacheBuilderPipeline(
                                logger=self._logger,
                                scrobble_utils=self._scrobble_utils,
                                embedder=self._embedder,
                                parameters=params,
                                test_mode=self._test_mode,
                                form_gold_standard=False,
                                cache_helper=self._cache_helper,
                                embedding_searcher=self._searcher
                            )
                            pipeline.build_pipeline()

                            # init context
                            ctx = AlgorithmContext(
                                original_scrobble_count=self._total_gold,
                                previous_pipe_description="None",
                                scrobble_data_frame=self._gold_df.drop(columns=["Predicted UUID", "Correct UUID"], errors="ignore"),
                                library_data_frame=self._library_df,
                                library_lookup_key_to_uuid=self._library_lookup,
                                uncertain_keys=[],
                                library_keys=self._library_keys,
                                library_index=self._library_index
                            )

                            # run pipeline
                            pipeline.flow(ctx)

                            # extract buckets
                            acc_df = ctx.auto_accepted_scrobbles if ctx.auto_accepted_scrobbles is not None else pd.DataFrame(columns=["__key", "__predicted_uuid"])
                            rej_df = ctx.auto_rejected_scrobbles if ctx.auto_rejected_scrobbles is not None else pd.DataFrame(columns=["__key"])
                            cof_df = ctx.confused_scrobbles if ctx.confused_scrobbles is not None else pd.DataFrame(columns=["__key"])

                            # merge with gold
                            acc_m = acc_df.merge(self._gold_df[["__key", "Correct UUID"]], on="__key", how="left")
                            rej_m = rej_df.merge(self._gold_df[["__key", "Correct UUID"]], on="__key", how="left")

                            # compute basic counts
                            TA = int(((acc_m["__predicted_uuid"] == acc_m["Correct UUID"]) & acc_m["Correct UUID"].notna()).sum())
                            FA = acc_m.shape[0] - TA
                            TR = int(rej_m["Correct UUID"].isna().sum())
                            FR = rej_m.shape[0] - TR
                            C = cof_df.shape[0]
                            coverage = (TA + TR) / self._total_gold

                            # DEBUG: log false accept/reject details
                            false_accepts = acc_m[acc_m["__predicted_uuid"] != acc_m["Correct UUID"]]
                            false_rejects = rej_m[rej_m["Correct UUID"].notna()]
                            if not false_accepts.empty or not false_rejects.empty:
                                self._logger.debug(
                                    f"Params(ca={ca},cr={cr},σ={sigma}) False Accepts: "
                                    f"{false_accepts[['__key','__predicted_uuid','Correct UUID']].to_dict('records')}; "
                                    f"False Rejects: {false_rejects['__key'].tolist()}",
                                    separator=self._separator
                                )

                            self._logger.info(
                                f"Params(ca={ca},cr={cr},σ={sigma},tok={tok},weight={embed_weight}) → "
                                f"TA={TA}, FA={FA}, TR={TR}, FR={FR}, C={C}, cov={coverage:.3f}",
                                separator=self._separator
                            )

                            if FA == 0 and FR == 0:
                                results.append({"params": params, "coverage": coverage, "C": C, "TA": TA, "TR": TR})

        # select best
        if not results:
            self._logger.error("No parameter set met FA==0 & FR==0; consider expanding grid.", separator=self._separator)
            return

        best = max(results, key=lambda r: (r["coverage"], -r["C"]))
        p: ScrobbleCacheAlgorithmParameters = best["params"]
        print(
            f"\nBest parameters (FA=0 & FR=0):\n"
            f"  accept_threshold = {p.confidence_accept_threshold}\n"
            f"  reject_threshold = {p.confidence_reject_threshold}\n"
            f"  token_threshold  = {p.token_accept_threshold}\n"
            f"  embed_weights    = {p.embed_weights}\n"
            f"  sigma            = {p.gaussian_sigma}\n"
            f"  top_k            = {p.top_k}\n\n"
            f"Coverage: {best['coverage']:.3f}, Confused: {best['C']}, TA={best['TA']}, TR={best['TR']}"
        )
