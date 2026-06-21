
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = 'AutoEDA - AI Data Preprocessing'
    APP_VERSION: str = '1.0.0'
    DEBUG: bool = False
    HOST: str = '0.0.0.0'
    PORT: int = 8000
    WORKERS: int = 1
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOADS_DIR: Path = BASE_DIR / 'uploads'
    OUTPUTS_DIR: Path = BASE_DIR / 'outputs'
    EDA_OUTPUT_DIR: Path = OUTPUTS_DIR / 'eda'
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: list[str] = ['csv', 'xlsx', 'json']
    OLLAMA_BASE_URL: str = 'http://localhost:11434'
    OLLAMA_MODEL: str = 'llama3'
    OLLAMA_TIMEOUT_SECONDS: int = 300
    OLLAMA_MAX_RETRIES: int = 3
    OLLAMA_RETRY_DELAY_SECONDS: int = 2
    AI_PLANNER_RESPONSE_TIMEOUT: int = 120
    AI_PLANNER_MAX_RETRIES: int = 3
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: Path = BASE_DIR / 'logs' / 'app.log'
    RANDOM_STATE: int = 42
    NAN_THRESHOLD_PERCENTAGE: float = 0.5
    DUPLICATE_THRESHOLD: float = 0.95
    NUMERICAL_IMPUTATION_METHOD: str = 'median'
    CATEGORICAL_IMPUTATION_METHOD: str = 'mode'
    OUTLIER_DETECTION_METHOD: str = 'iqr'
    IQR_MULTIPLIER: float = 1.5
    ZSCORE_THRESHOLD: float = 3.0
    CATEGORICAL_ENCODING_METHOD: str = 'one_hot'
    ONE_HOT_MAX_CATEGORIES: int = 10
    ENABLE_DATE_DECOMPOSITION: bool = True
    ENABLE_POLYNOMIAL_FEATURES: bool = False
    ENABLE_INTERACTION_FEATURES: bool = False
    NUMERICAL_SCALING_METHOD: str = 'standard'
    GENERATE_CORRELATION_HEATMAP: bool = True
    GENERATE_DISTRIBUTION_PLOTS: bool = True
    GENERATE_MISSING_VALUE_PLOTS: bool = True
    GENERATE_OUTLIER_PLOTS: bool = True
    PLOT_DPI: int = 300
    PLOT_FIGURE_SIZE: tuple[int, int] = (12, 6)
    PLOT_STYLE: str = 'seaborn-v0_8-darkgrid'
    MIN_READINESS_SCORE: int = 70

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = True

    def validate_directories(self) -> None:
        for directory in [self.UPLOADS_DIR, self.OUTPUTS_DIR, self.EDA_OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        log_dir = self.LOG_FILE.parent
        log_dir.mkdir(parents=True, exist_ok=True)
settings = Settings()
settings.validate_directories()