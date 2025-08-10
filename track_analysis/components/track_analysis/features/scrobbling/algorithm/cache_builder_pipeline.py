from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.compute_histogram_thresholds import \
    ComputeHistogramThresholds
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.extract_unique_entries import \
    ExtractUniqueEntries
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.filter_exact_matches import \
    FilterExactMatches
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.filter_manual_override import \
    FilterManualOverride
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.form_gold_standard import \
    FormGoldStandard
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.nearest_neighbour_search import \
    NearestNeighborSearch
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.report_uncertain_keys import \
    ReportUncertainKeys
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.status_report import StatusReport
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.store_in_cache import StoreInCache
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.validate_manual_override import \
    ValidateManualOverride
from track_analysis.components.track_analysis.features.scrobbling.embedding.embedding_searcher import EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobbling.model.scrabble_cache_algorithm_parameters import \
    ScrobbleCacheAlgorithmParameters
from track_analysis.components.track_analysis.features.scrobbling.utils.cache_helper import ScrobbleCacheHelper
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_utility import ScrobbleUtility
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class CacheBuilderPipeline(AbPipeline):
    def __init__(self, logger: HoornLogger, scrobble_utils: ScrobbleUtility, embedder: SentenceTransformer,
                 parameters: ScrobbleCacheAlgorithmParameters, cache_helper: ScrobbleCacheHelper, embedding_searcher: EmbeddingSearcher,
                 test_mode: bool, app_config: TrackAnalysisConfigurationModel, form_gold_standard: bool = False):
        super().__init__(logger, pipeline_descriptor="CacheBuilderPipeline")
        self._logger = logger
        self._scrobble_utils = scrobble_utils
        self._embedder = embedder
        self._cache_helper = cache_helper
        self._parameters = parameters
        self._test_mode = test_mode
        self._searcher = embedding_searcher
        self._manual_json_path = parameters.manual_override_path
        self._uncertain_keys_path = parameters.uncertain_keys_path
        self._form_gold_standard = form_gold_standard
        self._config = app_config

    def build_pipeline(self):
        status_report: StatusReport = StatusReport(self._logger)

        self._add_step(ExtractUniqueEntries(self._logger, self._scrobble_utils))
        self._add_step(status_report)
        self._add_step(ValidateManualOverride(self._logger, self._manual_json_path))
        self._add_step(FilterManualOverride(self._logger, self._searcher, self._parameters, self._config))
        self._add_step(status_report)
        self._add_step(FilterExactMatches(self._logger))
        self._add_step(status_report)
        self._add_step(NearestNeighborSearch(self._logger, self._parameters, test_mode=self._test_mode, embedding_searcher=self._searcher))
        self._add_step(StoreInCache(self._logger, self._cache_helper, test_mode=self._test_mode))
        self._add_step(status_report)
        self._add_step(ReportUncertainKeys(self._logger, uncertain_keys_path=self._uncertain_keys_path))
        self._add_step(ComputeHistogramThresholds(self._logger))

        if self._test_mode and self._form_gold_standard:
            self._add_step(FormGoldStandard(self._logger, self._parameters.max_gold_standard_entries, self._parameters.gold_standard_csv_path))
