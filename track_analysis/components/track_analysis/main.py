from pathlib import Path

from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.initialize_logger import get_logger

LOGGER: HoornLogger = get_logger()
USER_INPUT_HELPER: UserInputHelper = UserInputHelper(LOGGER)

def debug_extract_tags(tag_extractor: TagExtractor):
    def _always_true_validator(_: str) -> (bool, str):
        return True, ""

    # W:\media\music\[02] organized\[01] hq\Classical\Ludovico Einaudi\Elegy For The Arctic\01 Elegy for the Arctic.flac
    file_to_check: str = USER_INPUT_HELPER.get_user_input("Please enter the path to the audio file you want to extract:", str, validator_func=_always_true_validator)

    tag_extractor.extract(Path(file_to_check))

if __name__ == "__main__":
    tag_extractor: TagExtractor = TagExtractor(LOGGER)

    cmd: CommandLineInterface = CommandLineInterface(LOGGER)
    cmd.add_command(["extract_tags_debug", "etd"], "Debugs the extract tags function.", debug_extract_tags, arguments=[tag_extractor])
    cmd.start_listen_loop()
