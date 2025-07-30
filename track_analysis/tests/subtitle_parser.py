import pprint
import re
from typing import List, Dict, Tuple
from pathlib import Path


class SubtitleParser:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._validate_file()
        self.subtitles = self._parse_subtitles()

    def _validate_file(self):
        if not self.file_path.exists() or not self.file_path.is_file():
            raise FileNotFoundError(f"Subtitle file not found: {self.file_path}")
        if self.file_path.suffix.lower() != '.srt':
            raise ValueError("Only .srt subtitle files are supported.")

    def _parse_subtitles(self) -> List[Dict[str, str]]:
        with self.file_path.open(encoding='utf-8-sig') as f:
            content = f.read()

        blocks = re.split(r'\n\s*\n', content.strip())
        subtitles = []

        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) >= 3:
                timing_line = lines[1]
                text_lines = lines[2:]
                subtitles.append({
                    'timestamp': timing_line,
                    'text': ' '.join(text_lines)
                })

        return subtitles

    def get_keyword_timestamps(self, keywords: List[str]) -> Dict[str, List[str]]:
        keyword_map = {kw.lower(): [] for kw in keywords}

        for entry in self.subtitles:
            timestamp = entry['timestamp']
            text = entry['text'].lower()
            for keyword in keyword_map:
                if re.search(rf'\b{re.escape(keyword)}\b', text):
                    keyword_map[keyword].append(timestamp)

        return keyword_map


# Example usage:
if __name__ == "__main__":
    path = input("Enter the path of the subtitles file: ")
    keywords = ["pee", "piss", "urine", "peeing", "pissing", "urinating", "bathroom", "toilet", "wet", "dirty", "desperate", "write", "writing", "hold", "holding", "accident", "accidentally", "squirt", "squirting"]

    parser = SubtitleParser(path)
    results = parser.get_keyword_timestamps(keywords)

    for keyword, timestamps in results.items():
        if len(timestamps) < 1:
            continue

        print(f"'{keyword}' found at:")
        for time in timestamps:
            print(f"  {time}")

    path = path.replace(".srt", "__extracted_subs.txt")

    formatted = pprint.pformat(results)

    with open(path, "w", encoding="utf-8") as f:
        f.write(formatted)
