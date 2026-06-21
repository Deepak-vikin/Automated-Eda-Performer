from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ColumnProfile(BaseModel):
    name: str = Field(..., description='Column name')
    data_type: str = Field(..., description='Data type of the column')
    missing_count: int = Field(default=0, description='Number of missing values')
    missing_percentage: float = Field(default=0.0, description='Percentage of missing values')
    unique_count: int = Field(default=0, description='Number of unique values')
    duplicates_count: int = Field(default=0, description='Number of duplicate values')
    min_value: Optional[float] = Field(default=None, description='Minimum value')
    max_value: Optional[float] = Field(default=None, description='Maximum value')
    mean_value: Optional[float] = Field(default=None, description='Mean value')
    median_value: Optional[float] = Field(default=None, description='Median value')
    std_dev: Optional[float] = Field(default=None, description='Standard deviation')
    top_value: Optional[str] = Field(default=None, description='Most common value')
    top_frequency: Optional[int] = Field(default=None, description='Frequency of top value')
    is_datetime: bool = Field(default=False, description='Whether column is datetime')
    min_date: Optional[str] = Field(default=None, description='Earliest date')
    max_date: Optional[str] = Field(default=None, description='Latest date')

class DatasetProfile(BaseModel):
    filename: str = Field(..., description='Original filename')
    file_type: str = Field(..., description='File type (csv, xlsx, json)')
    upload_timestamp: str = Field(..., description='Upload timestamp')
    row_count: int = Field(..., description='Number of rows')
    column_count: int = Field(..., description='Number of columns')
    total_missing_values: int = Field(default=0, description='Total missing values')
    total_duplicate_rows: int = Field(default=0, description='Total duplicate rows')
    memory_usage_mb: float = Field(default=0.0, description='Memory usage in MB')
    columns: List[ColumnProfile] = Field(default_factory=list, description='Profile of each column')
    numerical_columns: List[str] = Field(default_factory=list, description='List of numerical columns')
    categorical_columns: List[str] = Field(default_factory=list, description='List of categorical columns')
    datetime_columns: List[str] = Field(default_factory=list, description='List of datetime columns')

class PreprocessingAction(BaseModel):
    column_name: str = Field(..., description='Column to apply action to')
    missing_strategy: Optional[str] = Field(default=None, description='Strategy for missing values: median, mean, mode, drop')
    outlier_strategy: Optional[str] = Field(default=None, description='Strategy for outliers: iqr, zscore, drop')
    encoding_strategy: Optional[str] = Field(default=None, description='Encoding strategy: one_hot, label, ordinal')
    scaling_strategy: Optional[str] = Field(default=None, description='Scaling strategy: standard, minmax, robust')
    datatype_conversion: Optional[str] = Field(default=None, description='Target data type for conversion')
    feature_removal: bool = Field(default=False, description='Whether to remove this feature')
    date_decomposition: Optional[List[str]] = Field(default=None, description='Date parts to decompose: year, month, day, weekday')
    normalization: bool = Field(default=False, description='Apply whitespace normalization')
    reasoning: str = Field(..., description='Reasoning for this action')

class PreprocessingPlan(BaseModel):
    plan_timestamp: str = Field(..., description='When the plan was generated')
    total_columns: int = Field(..., description='Total number of columns')
    actions: List[PreprocessingAction] = Field(default_factory=list, description='List of preprocessing actions')
    features_to_remove: List[str] = Field(default_factory=list, description='Features recommended for removal')
    target_column: Optional[str] = Field(default=None, description='Identified target column')
    summary: str = Field(..., description='Human-readable summary of the plan')

class CleaningAction(BaseModel):
    column_name: str = Field(..., description='Column affected')
    action_type: str = Field(..., description='Type of action: imputation, duplicate_removal, type_conversion, outlier_handling')
    action_details: str = Field(..., description='Details of what was done')
    rows_affected: int = Field(default=0, description='Number of rows affected')
    status: str = Field(default='success', description='Status: success, warning, error')

class EDAStatistics(BaseModel):
    column_name: str = Field(..., description='Column name')
    count: int = Field(..., description='Number of non-null values')
    missing: int = Field(..., description='Number of missing values')
    unique: int = Field(..., description='Number of unique values')
    mean: Optional[float] = Field(default=None)
    std: Optional[float] = Field(default=None)
    min: Optional[float] = Field(default=None)
    percentile_25: Optional[float] = Field(default=None)
    median: Optional[float] = Field(default=None)
    percentile_75: Optional[float] = Field(default=None)
    max: Optional[float] = Field(default=None)
    mode: Optional[str] = Field(default=None)
    mode_frequency: Optional[int] = Field(default=None)

class CorrelationMatrix(BaseModel):
    columns: List[str] = Field(..., description='Column names')
    matrix: List[List[float]] = Field(..., description='2D correlation matrix')

class OutlierAnalysis(BaseModel):
    column_name: str = Field(..., description='Column name')
    outlier_count: int = Field(..., description='Number of outliers')
    outlier_percentage: float = Field(..., description='Percentage of outliers')
    lower_bound: Optional[float] = Field(default=None, description='Lower bound for outliers')
    upper_bound: Optional[float] = Field(default=None, description='Upper bound for outliers')

class EDAResults(BaseModel):
    eda_timestamp: str = Field(..., description='When EDA was generated')
    descriptive_statistics: List[EDAStatistics] = Field(default_factory=list)
    correlation_matrix: Optional[CorrelationMatrix] = Field(default=None)
    missing_value_analysis: Dict[str, int] = Field(default_factory=dict)
    outlier_analysis: List[OutlierAnalysis] = Field(default_factory=list)
    generated_plots: List[str] = Field(default_factory=list, description='Paths to generated plot files')
    markdown_summary: str = Field(default='', description='Markdown summary of EDA')

class ReadinessReport(BaseModel):
    validation_timestamp: str = Field(..., description='When validation was performed')
    remaining_missing_values: int = Field(default=0)
    remaining_duplicates: int = Field(default=0)
    categorical_encoded: bool = Field(default=False)
    numerical_scaled: bool = Field(default=False)
    all_datatypes_valid: bool = Field(default=False)
    data_quality_checks: Dict[str, bool] = Field(default_factory=dict)
    readiness_score: int = Field(default=0, ge=0, le=100, description='ML readiness score 0-100')
    ready_for_training: bool = Field(default=False)
    issues: List[str] = Field(default_factory=list, description='List of remaining issues')
    recommendations: List[str] = Field(default_factory=list, description='Recommendations for improvement')

class ProcessingState(BaseModel):
    dataset_path: str = Field(default='', description='Path to uploaded dataset')
    original_dataframe: Optional[str] = Field(default=None, description='Serialized original dataframe (for state storage)')
    dataset_profile: Optional[DatasetProfile] = Field(default=None)
    preprocessing_plan: Optional[PreprocessingPlan] = Field(default=None)
    cleaning_actions: List[CleaningAction] = Field(default_factory=list)
    cleaned_dataframe: Optional[str] = Field(default=None, description='Serialized cleaned dataframe')
    feature_engineering_log: List[Dict[str, Any]] = Field(default_factory=list)
    processed_dataframe: Optional[str] = Field(default=None, description='Serialized processed dataframe')
    eda_results: Optional[EDAResults] = Field(default=None)
    readiness_report: Optional[ReadinessReport] = Field(default=None)
    generated_files: List[str] = Field(default_factory=list, description='Paths to all generated output files')
    execution_logs: List[str] = Field(default_factory=list, description='Execution log messages')
    session_id: str = Field(..., description='Unique session identifier')
    created_at: str = Field(..., description='Session creation timestamp')
    updated_at: str = Field(..., description='Last update timestamp')

class UploadResponse(BaseModel):
    success: bool = Field(..., description='Whether upload was successful')
    message: str = Field(..., description='Status message')
    session_id: str = Field(..., description='Unique session ID for this processing')
    filename: str = Field(..., description='Uploaded filename')
    file_size_mb: float = Field(..., description='File size in MB')

    class Config:
        json_schema_extra = {'example': {'success': True, 'message': 'File uploaded successfully', 'session_id': 'abc123def456', 'filename': 'data.csv', 'file_size_mb': 5.2}}

class ProcessRequest(BaseModel):
    session_id: str = Field(..., description='Session ID from upload')

    class Config:
        json_schema_extra = {'example': {'session_id': 'abc123def456'}}

class ProcessResponse(BaseModel):
    success: bool = Field(..., description='Whether processing was successful')
    message: str = Field(..., description='Status message')
    session_id: str = Field(..., description='Session ID')
    profile: Optional[DatasetProfile] = Field(default=None)
    plan: Optional[PreprocessingPlan] = Field(default=None)

class ResultsResponse(BaseModel):
    success: bool = Field(..., description='Whether operation was successful')
    session_id: str = Field(..., description='Session ID')
    profile: Optional[DatasetProfile] = Field(default=None)
    plan: Optional[PreprocessingPlan] = Field(default=None)
    cleaning_actions: List[CleaningAction] = Field(default_factory=list)
    eda_results: Optional[EDAResults] = Field(default=None)
    readiness_report: Optional[ReadinessReport] = Field(default=None)
    generated_files: List[str] = Field(default_factory=list)
    execution_logs: List[str] = Field(default_factory=list)

class DownloadResponse(BaseModel):
    success: bool = Field(..., description='Whether download is available')
    message: str = Field(..., description='Status message')
    file_path: Optional[str] = Field(default=None, description='Path to file for download')
    file_name: Optional[str] = Field(default=None, description='Filename for download')

class HealthResponse(BaseModel):
    status: str = Field(default='healthy', description='Service status')
    timestamp: str = Field(..., description='Current timestamp')
    version: str = Field(default='1.0.0', description='API version')

    class Config:
        json_schema_extra = {'example': {'status': 'healthy', 'timestamp': '2024-01-15T10:30:00', 'version': '1.0.0'}}