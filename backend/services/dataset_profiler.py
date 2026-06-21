import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from backend.models.schemas import ColumnProfile, DatasetProfile
from backend.utils.exceptions import DatasetLoadException, DatasetProfilingException
from backend.config import settings
logger = logging.getLogger(__name__)

class DatasetProfiler:
    LOADERS = {'csv': pd.read_csv, 'xlsx': pd.read_excel, 'xls': pd.read_excel, 'json': pd.read_json}

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[Path] = None
        self.file_type: Optional[str] = None

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f'File not found: {file_path}')
            file_ext = file_path_obj.suffix.lower().lstrip('.')
            if file_ext not in self.LOADERS:
                raise ValueError(f"Unsupported file type: {file_ext}. Supported types: {', '.join(self.LOADERS.keys())}")
            logger.info(f'Loading dataset from {file_path} (type: {file_ext})')
            loader = self.LOADERS[file_ext]
            if file_ext in ['xlsx', 'xls']:
                df = loader(file_path, sheet_name=0)
            else:
                df = loader(file_path)
            if df.index.name is not None:
                df = df.reset_index(drop=True)
            self.df = df
            self.file_path = file_path_obj
            self.file_type = file_ext
            logger.info(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')
            return df
        except Exception as e:
            logger.error(f'Failed to load dataset: {str(e)}')
            raise DatasetLoadException(f'Failed to load dataset: {str(e)}') from e

    def _profile_column(self, column_name: str, column_data: pd.Series) -> ColumnProfile:
        data_type = str(column_data.dtype)
        missing_count = column_data.isna().sum()
        missing_percentage = missing_count / len(column_data) * 100
        unique_count = column_data.nunique()
        duplicates_count = len(column_data) - column_data.nunique()
        profile = ColumnProfile(name=column_name, data_type=data_type, missing_count=int(missing_count), missing_percentage=round(missing_percentage, 2), unique_count=int(unique_count), duplicates_count=int(duplicates_count))
        if pd.api.types.is_numeric_dtype(column_data):
            non_null = column_data.dropna()
            if len(non_null) > 0:
                profile.min_value = float(non_null.min())
                profile.max_value = float(non_null.max())
                profile.mean_value = float(non_null.mean())
                profile.median_value = float(non_null.median())
                profile.std_dev = float(non_null.std())
        elif pd.api.types.is_object_dtype(column_data) or pd.api.types.is_categorical_dtype(column_data):
            value_counts = column_data.value_counts()
            if len(value_counts) > 0:
                profile.top_value = str(value_counts.index[0])
                profile.top_frequency = int(value_counts.iloc[0])
        try:
            if pd.api.types.is_datetime64_any_dtype(column_data):
                profile.is_datetime = True
                non_null = column_data.dropna()
                if len(non_null) > 0:
                    profile.min_date = str(non_null.min())
                    profile.max_date = str(non_null.max())
        except Exception:
            pass
        return profile

    def _detect_column_types(self) -> Tuple[List[str], List[str], List[str]]:
        numerical_cols = []
        categorical_cols = []
        datetime_cols = []
        for col in self.df.columns:
            col_data = self.df[col]
            try:
                if pd.api.types.is_datetime64_any_dtype(col_data):
                    datetime_cols.append(col)
                    continue
            except Exception:
                pass
            if pd.api.types.is_numeric_dtype(col_data):
                numerical_cols.append(col)
            elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                categorical_cols.append(col)
        return (numerical_cols, categorical_cols, datetime_cols)

    def profile_dataset(self) -> DatasetProfile:
        if self.df is None:
            raise DatasetProfilingException('No dataset loaded. Call load_dataset() first.')
        try:
            logger.info('Starting dataset profiling...')
            row_count = len(self.df)
            column_count = len(self.df.columns)
            total_missing_values = int(self.df.isna().sum().sum())
            total_duplicate_rows = int(self.df.duplicated().sum())
            memory_usage_mb = float(self.df.memory_usage(deep=True).sum() / 1024 ** 2)
            (numerical_cols, categorical_cols, datetime_cols) = self._detect_column_types()
            columns_profile = []
            for col in self.df.columns:
                try:
                    col_profile = self._profile_column(col, self.df[col])
                    columns_profile.append(col_profile)
                except Exception as e:
                    logger.warning(f'Failed to profile column {col}: {str(e)}')
            profile = DatasetProfile(filename=self.file_path.name if self.file_path else 'unknown', file_type=self.file_type or 'unknown', upload_timestamp=datetime.now().isoformat(), row_count=row_count, column_count=column_count, total_missing_values=total_missing_values, total_duplicate_rows=total_duplicate_rows, memory_usage_mb=round(memory_usage_mb, 2), columns=columns_profile, numerical_columns=numerical_cols, categorical_columns=categorical_cols, datetime_columns=datetime_cols)
            logger.info(f'Dataset profiling complete: {column_count} columns, {row_count} rows')
            return profile
        except DatasetProfilingException:
            raise
        except Exception as e:
            logger.error(f'Dataset profiling failed: {str(e)}')
            raise DatasetProfilingException(f'Failed to profile dataset: {str(e)}') from e

    def get_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            raise DatasetProfilingException('No dataset loaded')
        return self.df.copy()