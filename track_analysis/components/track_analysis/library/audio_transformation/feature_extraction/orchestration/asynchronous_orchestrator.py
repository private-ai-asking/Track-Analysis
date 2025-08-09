import asyncio
from typing import List, Dict, Any

from track_analysis.components.track_analysis.features.data_generation.processors.main_feature_processor import \
    TrackProcessingResult
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider, ProviderResult, ProviderProcessingStatistics
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.orchestration.orchestrator_interface import \
    OrchestratorInterface
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.provider_dependency_management.dependency_resolver import \
    DependencyResolver


class AsynchronousOrchestrator(OrchestratorInterface):
    """
    Executes the feature dependency graph with maximum concurrency using asyncio.

    This orchestrator turns the execution plan into a graph of awaitable tasks,
    allowing I/O and CPU-bound features to be processed in parallel,
    dramatically improving resource utilization for a single track.
    """
    def __init__(self, providers: List[AudioDataFeatureProvider]):
        self._resolver = DependencyResolver(providers)
        self._providers_map = {p.__class__.__name__: p for p in providers}

    async def process_track(
            self,
            initial_data: Dict[AudioDataFeature, Any],
            features_to_calculate: List[AudioDataFeature]
    ) -> TrackProcessingResult:
        """
        Processes a track by creating a dynamic task graph and executing it with asyncio.
        """
        execution_plan, required_base = self._resolver.resolve(features_to_calculate)
        feature_tasks: Dict[AudioDataFeature, asyncio.Task] = {}

        # 1. Prime the task graph with initial data by creating already-completed tasks
        for feature, value in initial_data.items():
            async def completed_coro(val=value):
                return {feature: val}
            feature_tasks[feature] = asyncio.create_task(completed_coro())

        # 2. Create tasks for all providers in the execution plan.
        for provider in execution_plan:
            task = asyncio.create_task(
                self._run_provider_after_dependencies(provider, feature_tasks)
            )
            for out_feature in self._get_provider_outputs(provider):
                feature_tasks[out_feature] = task

        # 3. Wait for only the final, requested features to complete.
        final_tasks = [feature_tasks[f] for f in features_to_calculate if f in feature_tasks]
        await asyncio.gather(*final_tasks)

        # 4. Collect all results and statistics
        final_features = initial_data.copy()
        all_stats: Dict[AudioDataFeatureProvider, ProviderProcessingStatistics] = {}

        for provider in execution_plan:
            provider_task = feature_tasks[self._get_provider_outputs(provider)[0]]
            if provider_task.done():
                provider_result = provider_task.result()
                final_features.update(provider_result.retrieved_features)
                all_stats[provider] = provider_result.statistics

        return TrackProcessingResult(retrieved_features=final_features, provider_stats=all_stats)

    @staticmethod
    async def _run_provider_after_dependencies(
            provider: AudioDataFeatureProvider,
            feature_tasks: Dict[AudioDataFeature, asyncio.Task]
    ) -> ProviderResult:
        """
        A helper coroutine that waits for dependencies, runs a provider, and returns its result.
        """
        # Gather the tasks for all dependencies
        dependency_tasks = [feature_tasks[dep] for dep in provider.dependencies]

        # Wait for all dependency tasks to complete
        dependency_results_list = await asyncio.gather(*dependency_tasks)

        # Build the input data dictionary for the provider
        input_data = {}
        for result_dict in dependency_results_list:
            input_data.update(result_dict)

        # Finally, run the provider's async `provide` method
        return await provider.provide(input_data)

    @staticmethod
    def _get_provider_outputs(provider: AudioDataFeatureProvider) -> List[AudioDataFeature]:
        outputs = provider.output_features
        return outputs if isinstance(outputs, list) else [outputs]
