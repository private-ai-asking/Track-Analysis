import dataclasses
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar

from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel, ScrobbleLinkerConfig, PathConfig, AdditionalConfiguration, DevelopmentConfig

T = TypeVar('T')

class ConfigurationLoader:
    """Loads the track analysis app configuration."""
    def __init__(self, configuration_path: Path):
        self._configuration_path = configuration_path
        self._placeholder_pattern = re.compile(r"\{([^}]+)}")

        self._validate_path(configuration_path)

    @staticmethod
    def _validate_path(configuration_path: Path) -> None:
        if not configuration_path.exists():
            raise ValueError(f"Configuration '{configuration_path}' does not exist.")
        if not configuration_path.suffix == ".json":
            raise ValueError(f"Configuration '{configuration_path}' is not a .json file.")

    def get_configuration(self) -> TrackAnalysisConfigurationModel:
        try:
            with open(self._configuration_path, 'r') as f:
                raw_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON configuration: {e}")

        resolved_config = self._resolve_placeholders(raw_config)
        path_config_data = resolved_config.get("Path Configuration", {})
        scrobble_config_data = resolved_config.get("Scrobble Linking Configuration", {})
        additional_config_data = resolved_config.get("Additional Configuration", {})
        development_config = resolved_config.get("Development Configuration", {})

        path_config = self._create_config_from_dict(path_config_data, PathConfig)
        scrobble_linker_config = self._create_config_from_dict(scrobble_config_data, ScrobbleLinkerConfig)
        additional_config = self._create_config_from_dict(additional_config_data, AdditionalConfiguration)
        development_config = self._create_config_from_dict(development_config, DevelopmentConfig)

        return TrackAnalysisConfigurationModel(paths=path_config, scrobble_linker=scrobble_linker_config, additional_config=additional_config, development=development_config)

    # noinspection t
    @staticmethod
    def _create_config_from_dict(data: Dict[str, Any], model_class: Type[T]) -> T:
        """
        Generic factory for creating a dataclass instance from a dictionary.
        It inspects the model to find required fields and ignores extras.
        """
        # 1. Get all expected field names from the dataclass model
        expected_fields = {f.name for f in dataclasses.fields(model_class)}
        constructor_args = {}

        # 2. Iterate through JSON data and match keys to model fields
        for key, value in data.items():
            field_name = key.replace('-', '_')
            if field_name in expected_fields:
                # 3. Handle type casting if necessary (e.g., for Path)
                field_type = model_class.__annotations__[field_name]
                if field_type is Path:
                    constructor_args[field_name] = Path(value)
                else:
                    constructor_args[field_name] = value

        # 4. Create the dataclass instance with only the valid arguments
        try:
            return model_class(**constructor_args)
        except TypeError as e:
            raise ValueError(f"Mismatched arguments for {model_class.__name__}. Check your JSON and model definition. Error: {e}")

    # noinspection t
    def _resolve_placeholders(self, config: Dict[str, Any]) -> Dict[str, Any]:
        made_substitution = True
        while made_substitution:
            made_substitution = False
            for section, settings in config.items():
                if isinstance(settings, dict):
                    for key in settings:
                        if self._process_setting(config, key, settings):
                            made_substitution = True
        return config

    def _process_setting(self, config: Dict[str, Any], key: str, settings: Dict) -> bool:
        value = settings[key]
        placeholder_ref = self._get_placeholder_ref(value)
        if placeholder_ref:
            new_value = self._attempt_substitution(config, value, placeholder_ref)
            if new_value is not None:
                settings[key] = new_value
                return True
        return False

    def _get_placeholder_ref(self, value: str) -> Optional[str]:
        if not isinstance(value, str):
            return None
        match = self._placeholder_pattern.search(value)
        return match.group(1) if match else None

    def _attempt_substitution(self, config: Dict[str, Any], value: str, placeholder_ref: str) -> Optional[str]:
        try:
            ref_parts = placeholder_ref.split('/')
            resolved_value = config
            for part in ref_parts:
                resolved_value = resolved_value[part]

            if not self._placeholder_pattern.search(str(resolved_value)):
                return value.replace(f"{{{placeholder_ref}}}", str(resolved_value))
        except (KeyError, ValueError, TypeError):
            pass
        return None

