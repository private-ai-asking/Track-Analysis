from typing import List

from track_analysis.components.md_common_python.py_common.misc import DirectoryTreeConfig, DirectoryTreeGenerator
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.constants import ROOT


class SaveDirectoryTreeCommand(CommandExecutionModel):
    @property
    def default_command_keys(self) -> List[str]:
        return ["save_directory_tree", "sdt"]

    @property
    def command_description(self) -> str:
        return "Saves the developer's directory tree."

    def execute(self, arguments: T) -> P:
        config: DirectoryTreeConfig = DirectoryTreeConfig()
        generator: DirectoryTreeGenerator = DirectoryTreeGenerator(config)

        output_path = ROOT / "directory_tree.txt"
        root_path = ROOT / "track_analysis" / "components" / "track_analysis"
        generator.generate(root_path, output_path)
