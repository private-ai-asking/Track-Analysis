from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Dict, Any, Tuple, List, Union

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider, ProviderProcessingStatistics
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.model.execution_plan import \
    ExecutionPlan, ExecutionStage
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.model.provider_output import \
    ProviderOutput
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.model.track_processing_result import \
    TrackProcessingResult
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.validation.feature_data_validator import \
    FeatureDataValidator


class PlanExecutor:
    """Handles the execution of feature provider stages for a single track."""
    def __init__(self, feature_threads_num: int):
        self._feature_threads_num = feature_threads_num

    def execute_plan(
            self,
            initial_data: Dict[AudioDataFeature, Any],
            execution_plan: ExecutionPlan
    ) -> TrackProcessingResult:
        """Iterates through a specific execution plan and runs each feature provider."""
        results = initial_data.copy()
        provider_stats: Dict[AudioDataFeatureProvider, ProviderProcessingStatistics] = {}

        executor = ThreadPoolExecutor(max_workers=self._feature_threads_num)

        with executor:
            for stage in execution_plan.stages:
                stage_results, stage_stats = self._execute_stage(executor, stage, results)
                results.update(stage_results)
                provider_stats.update(stage_stats)

        return TrackProcessingResult(retrieved_features=results, provider_stats=provider_stats)

    def _execute_stage(self, executor: ThreadPoolExecutor, stage: ExecutionStage, input_data: Dict[AudioDataFeature, Any]) -> Tuple[Dict[AudioDataFeature, Any], Dict[AudioDataFeatureProvider, ProviderProcessingStatistics]]:
        active_futures: List[Future] = [
            executor.submit(self._get_provider_output, provider, input_data)
            for provider in stage.providers
        ]

        stage_output: List[ProviderOutput] = []

        for future in as_completed(active_futures):
            result: ProviderOutput = future.result()
            stage_output.append(result)

        results: Dict[AudioDataFeature, Any] = {}
        provider_stats: Dict[AudioDataFeatureProvider, ProviderProcessingStatistics] = {}

        for provider_output in stage_output:
            self._handle_provider_output(provider_output, results, provider_stats)

        return results, provider_stats

    def _handle_provider_output(self,
                                output: ProviderOutput,
                                results: Dict[AudioDataFeature, Any],
                                provider_stats: Dict[AudioDataFeatureProvider, ProviderProcessingStatistics]) -> None:
        retrieved_features = output.provided_data.retrieved_features
        provider_stats[output.provider] = output.provided_data.statistics

        FeatureDataValidator.validate_output(output.provider.output_features, retrieved_features, output.provider_name)
        self._update_results(results, output.provider.output_features, retrieved_features)

    @staticmethod
    def _get_provider_output(feature_provider: AudioDataFeatureProvider,
                             results: Dict[AudioDataFeature, Any]) -> ProviderOutput:
        provider_name: str = feature_provider.__class__.__name__
        FeatureDataValidator.validate_dependencies(feature_provider.dependencies, results, provider_name)

        input_data = {dep: results[dep] for dep in feature_provider.dependencies}
        provided_data = feature_provider.provide(input_data)
        return ProviderOutput(
            provider=feature_provider,
            provided_data=provided_data,
            provider_name=provider_name,
        )

    @staticmethod
    def _update_results(
            results: Dict[AudioDataFeature, Any],
            output_features: Union[AudioDataFeature, List[AudioDataFeature]],
            calculated_data: Any,
    ) -> None:
        """Updates the results dictionary for single or multi-output calculators."""
        if isinstance(calculated_data, dict):
            results.update(calculated_data)
        else:
            results[output_features] = calculated_data
