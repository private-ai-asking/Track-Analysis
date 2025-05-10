from pathlib import Path

from track_analysis.components.md_common_python.py_common.component_registration import ComponentRegistration
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface


class RegistrationTest(TestInterface):
    def __init__(self, logger: HoornLogger, register_component: ComponentRegistration):
        super().__init__(logger, is_child=True)
        self._separator: str = "RegistrationTest"
        self._registration: ComponentRegistration = register_component
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def test(self, **kwargs) -> None:
        registration_path: Path = Path("X:\\Track Analysis\\track_analysis\components\\track_analysis\\registration.json")
        signature_path: Path = Path("X:\\Track Analysis\\track_analysis\components\\track_analysis\\component_signature.json")

        self._registration.register_component(registration_path, signature_path)
