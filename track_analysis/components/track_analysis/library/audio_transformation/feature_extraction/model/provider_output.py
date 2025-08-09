import dataclasses

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider, ProviderResult


@dataclasses.dataclass(frozen=True)
class ProviderOutput:
    provider: AudioDataFeatureProvider
    provided_data: ProviderResult
    provider_name: str
