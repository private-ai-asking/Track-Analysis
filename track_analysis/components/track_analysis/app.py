from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.track_analysis.constants import ROOT_MUSIC_LIBRARY, OUTPUT_DIRECTORY
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.pipeline.pipeline import Pipeline
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class App:
    def __init__(self, logger: HoornLogger):
        self._user_input_helper: UserInputHelper = UserInputHelper(logger)
        self._tag_extractor: TagExtractor = TagExtractor(logger)
        self._file_handler: FileHandler = FileHandler()
        self._logger = logger

    def run(self):
        cmd: CommandLineInterface = CommandLineInterface(self._logger)
        cmd.add_command(["extract_tags_debug", "etd"], "Debugs the extract tags function.", self._debug_extract_tags)
        cmd.add_command(["make_csv", "mc"], "Makes a CSV file from the extracted metadata.", self._make_csv)
        cmd.start_listen_loop()

    def _add_album_cost(self, album_costs: List[AlbumCostModel], title: str, cost: float) -> List[AlbumCostModel]:
        album_costs.append(AlbumCostModel(Album_Title=title, Album_Cost=cost))
        return album_costs

    def _debug_extract_tags(self):
        def _always_true_validator(_: str) -> (bool, str):
            return True, ""

        # W:\media\music\[02] organized\[01] hq\Classical\Ludovico Einaudi\Elegy For The Arctic\01 Elegy for the Arctic.flac
        # W:\media\music\[02] organized\[02] lq\CCM\Champion\15 - Beckah Shae - Incorruptible (David Thulin remix).flac
        file_to_check: str = self._user_input_helper.get_user_input("Please enter the path to the audio file you want to extract:", str, validator_func=_always_true_validator)

        result: AudioInfo =self._tag_extractor.extract(Path(file_to_check))

        for metadata_item in result.metadata:
            self._logger.info(f"{metadata_item.header} - {metadata_item.description}: {metadata_item.value}")

    def _make_csv(self):
        album_costs = []

        album_costs = self._add_album_cost(album_costs, "Classical Best", 10.49)
        album_costs = self._add_album_cost(album_costs, "The Hours (Music from the Motion Picture)", 12.49)
        album_costs = self._add_album_cost(album_costs, "Old Friends New Friends", 20.39)

        album_costs = self._add_album_cost(album_costs, "Musica baltica", 15.19)
        album_costs = self._add_album_cost(album_costs, "Prehension", 16.29)
        album_costs = self._add_album_cost(album_costs, "Solipsism", 13.59)

        album_costs = self._add_album_cost(album_costs, "The Blue Notebooks (20 Year Edition)", 16.29)
        album_costs = self._add_album_cost(album_costs, "In a Time Lapse", 16.29)
        album_costs = self._add_album_cost(album_costs, "Una mattina", 16.29)
        album_costs = self._add_album_cost(album_costs, "Eden Roc", 16.29)
        album_costs = self._add_album_cost(album_costs, "I Giorni", 16.29)
        album_costs = self._add_album_cost(album_costs, "Le onde", 16.29)

        album_costs = self._add_album_cost(album_costs, "Lead Thou Me On: Hymns and Inspiration", 9.49)
        album_costs = self._add_album_cost(album_costs, "Lux", 13.59)
        album_costs = self._add_album_cost(album_costs, "Eventide", 13.59)
        album_costs = self._add_album_cost(album_costs, "Light and Gold", 30.79)
        album_costs = self._add_album_cost(album_costs, "De la taberna a la Corte", 12.59)
        album_costs = self._add_album_cost(album_costs, "Edvard Grieg a capella", 10.49)
        album_costs = self._add_album_cost(album_costs, "Edvard Grieg - Essential Orchestral Works", 5.79)
        album_costs = self._add_album_cost(album_costs, "The Young Beethoven", 10.79)
        album_costs = self._add_album_cost(album_costs, "The Young Messiah", 10.79)
        album_costs = self._add_album_cost(album_costs, "Ode To Joy", 10.79)
        album_costs = self._add_album_cost(album_costs, "Satie: Gymnopédies; Gnossienne", 8.59)
        album_costs = self._add_album_cost(album_costs, "The Very Best of Arvo Pärt", 9.29)
        album_costs = self._add_album_cost(album_costs, "Elegy for the Arctic", 1.99)
        album_costs = self._add_album_cost(album_costs, "Alina", 13.59)
        album_costs = self._add_album_cost(album_costs, "Divenire", 16.29)
        album_costs = self._add_album_cost(album_costs, "Elements", 20.69)
        album_costs = self._add_album_cost(album_costs, "Memoryhouse", 10.79)

        album_costs = self._add_album_cost(album_costs, "Halfway Tree", 13.59)
        album_costs = self._add_album_cost(album_costs, "Welcome to Jamrock", 13.59)
        album_costs = self._add_album_cost(album_costs, "Mr. Marley", 13.59)
        album_costs = self._add_album_cost(album_costs, "Distant Relatives", 13.59)
        album_costs = self._add_album_cost(album_costs, "Rapture", 6.49)
        album_costs = self._add_album_cost(album_costs, "Stony Hill", 23.99)
        album_costs = self._add_album_cost(album_costs, "A Matter of Time", 8.99)

        pipeline_context = PipelineContextModel(
            source_dir=ROOT_MUSIC_LIBRARY,
            output_file_path=OUTPUT_DIRECTORY.joinpath("data.csv"),
            album_costs=album_costs
        )

        pipeline = Pipeline(self._logger, self._file_handler, self._tag_extractor)
        pipeline.build_pipeline()
        pipeline.flow(pipeline_context)
