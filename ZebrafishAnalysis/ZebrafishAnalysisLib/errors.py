class AnalysisInputError(ValueError):
    """Raised by ZebrafishAnalysisLogic when analysis inputs fail validation."""


class MRMLAdapterError(RuntimeError):
    """Raised when MRML scene integration fails after a successful analysis."""
