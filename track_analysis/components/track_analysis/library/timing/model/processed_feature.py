from dataclasses import dataclass


@dataclass
class ProcessedFeature:
    name: str = ""
    total_time: float = 0.0
    wait_time: float = 0.0
    process_time: float = 0.0
    call_count: int = 0
    avg_time_ms: float = 0.0
