import dataclasses
import hashlib
from typing import List

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider


@dataclasses.dataclass(frozen=True)
class ExecutionStage:
    stage_number: int
    providers: List[AudioDataFeatureProvider]


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    stages: List[ExecutionStage] = dataclasses.field(default_factory=list)

    def get_hash(self) -> str:
        stage_ids: List[str] = []

        for stage in self.stages:
            provider_names = sorted([p.__class__.__name__ for p in stage.providers])
            stage_ids.append('_'.join(provider_names) + f"_{stage.stage_number}")

        generated_hash = hashlib.sha256('|'.join(stage_ids).encode('utf-8')).hexdigest()
        return generated_hash
