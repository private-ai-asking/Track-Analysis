import dataclasses


@dataclasses.dataclass(frozen=True)
class ParallelismStats:
    """Holds the calculated statistics for a parallel workload."""
    wall_clock_time: float
    worker_count: int
    speedup_factor: float
    worker_utilization_percent: float
    total_overhead_time: float
    avg_throughput: float
    min_task_time: float
    max_task_time: float
    mean_task_time: float
    median_task_time: float
    stdev_task_time: float
    sequential_time: float
    task_count: int
