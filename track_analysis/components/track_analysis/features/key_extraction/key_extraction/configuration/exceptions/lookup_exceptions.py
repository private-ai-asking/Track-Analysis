class MalformedLabelError(Exception):
    """Raised when a state label cannot be split into tonic and mode."""
    def __init__(self, label: str):
        super().__init__(f"Invalid state label format: '{label}'")
        self.label = label


class UnknownTonicError(Exception):
    """Raised when the parsed tonic is not found in the index map."""
    def __init__(self, tonic: str, label: str):
        super().__init__(f"Unknown tonic '{tonic}' in '{label}'")
        self.tonic = tonic
        self.label = label
