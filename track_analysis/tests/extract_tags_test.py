from pathlib import Path

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.features.data_generation.model.audio_info import AudioInfo


class ExtractTagsTest(TestInterface):
    def __init__(self, logger: HoornLogger, user_input_helper: UserInputHelper, tag_extractor: TagExtractor):
        self._user_input_helper = user_input_helper
        self._tag_extractor = tag_extractor
        super().__init__(logger, is_child=True)

    def test(self, **kwargs) -> None:
        # W:\media\music\[02] organized\[01] hq\Classical\Ludovico Einaudi\Elegy For The Arctic\01 Elegy for the Arctic.flac
        # W:\media\music\[02] organized\[02] lq\CCM\Champion\15 - Beckah Shae - Incorruptible (David Thulin remix).flac
        file_to_check: str = self._user_input_helper.get_user_input("Please enter the path to the audio file you want to extract:", str, validator_func=lambda s: (True, ""))

        result: AudioInfo =self._tag_extractor.extract(Path(file_to_check))

        for metadata_item in result.metadata:
            self._logger.info(f"{metadata_item.header} - {metadata_item.description}: {metadata_item.value}")
