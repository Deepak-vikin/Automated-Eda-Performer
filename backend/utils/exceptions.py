class AutoEDAException(Exception):
    pass

class DatasetUploadException(AutoEDAException):
    pass

class DatasetLoadException(AutoEDAException):
    pass

class DatasetProfilingException(AutoEDAException):
    pass

class OllamaException(AutoEDAException):
    pass

class AIPreprocessingPlanException(AutoEDAException):
    pass

class PreprocessingException(AutoEDAException):
    pass

class FeatureEngineeringException(AutoEDAException):
    pass

class EDAGenerationException(AutoEDAException):
    pass

class ReadinessValidationException(AutoEDAException):
    pass

class OutputGenerationException(AutoEDAException):
    pass

class ValidationException(AutoEDAException):
    pass

class GraphExecutionException(AutoEDAException):
    pass