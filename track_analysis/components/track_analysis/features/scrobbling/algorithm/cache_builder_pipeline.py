from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import AbPipeline
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.extract_unique_entries import \
    ExtractUniqueEntries
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.filter_exact_matches import \
    FilterExactMatches
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.nearest_neighbour_search import \
    NearestNeighbourSearch
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.report_uncertain_keys import \
    ReportUncertainKeys
from track_analysis.components.track_analysis.features.scrobbling.algorithm.pipes.status_report import StatusReport
from track_analysis.components.track_analysis.features.scrobbling.model.scrabble_cache_algorithm_parameters import \
    ScrobbleCacheAlgorithmParameters
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class CacheBuilderPipeline(AbPipeline):
    def __init__(self, logger: HoornLogger, scrobble_utils: ScrobbleUtility, embedder: SentenceTransformer,
                 scorer: SimilarityScorer, parameters: ScrobbleCacheAlgorithmParameters, test_mode: bool):
        super().__init__(logger, pipeline_descriptor="CacheBuilderPipeline")
        self._logger = logger
        self._scrobble_utils = scrobble_utils
        self._embedder = embedder
        self._scorer = scorer
        self._parameters = parameters
        self._test_mode = test_mode

    def build_pipeline(self):
        status_report: StatusReport = StatusReport(self._logger)

        self._add_step(ExtractUniqueEntries(self._logger, self._scrobble_utils))
        self._add_step(status_report)
        self._add_step(FilterExactMatches(self._logger, self._scrobble_utils))
        self._add_step(status_report)
        self._add_step(NearestNeighbourSearch(self._logger, self._scrobble_utils, self._embedder, self._scorer, self._parameters, self._test_mode))
        self._add_step(status_report)
        self._add_step(ReportUncertainKeys(self._logger))
