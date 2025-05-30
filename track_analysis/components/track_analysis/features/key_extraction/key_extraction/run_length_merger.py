from typing import List, Tuple, Dict


class RunLengthMerger:
    @staticmethod
    def merge(
            times: List[Tuple[float, float]],
            states: List[int],
            labels: List[str]
    ) -> List[Dict]:
        """
        Collapse consecutive identical states into runs with start/end and label.
        times: list of (start, end) per segment
        states: list of state indices per segment
        labels: list of state names
        """
        merged = []
        if not states:
            return merged
        start_idx = 0
        current_state = states[0]
        for i in range(1, len(states)):
            if states[i] != current_state:
                merged.append({
                    'start': times[start_idx][0],
                    'end': times[i - 1][1],
                    'state': labels[current_state],
                    'idx': start_idx
                })
                start_idx = i
                current_state = states[i]
        # last run
        merged.append({
            'start': times[start_idx][0],
            'end': times[-1][1],
            'state': labels[current_state],
            'idx': start_idx
        })
        return merged
