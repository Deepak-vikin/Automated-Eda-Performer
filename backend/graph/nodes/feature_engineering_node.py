import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from backend.config import settings
from backend.graph.state import PreprocessingGraphState
from backend.models.schemas import PreprocessingAction
from backend.utils.exceptions import FeatureEngineeringException
logger = logging.getLogger(__name__)

class FeatureEngineeringExecutor:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.engineering_log: List[Dict[str, Any]] = []
        self.encoder_configs: Dict[str, Any] = {}
        self.scaler_configs: Dict[str, Any] = {}
        self.original_shape = df.shape

    def _log_operation(self, operation: str, column: str, details: str, columns_added: int=0, columns_removed: int=0, status: str='success') -> None:
        entry = {'operation': operation, 'column': column, 'details': details, 'columns_added': columns_added, 'columns_removed': columns_removed, 'status': status, 'timestamp': datetime.now().isoformat()}
        self.engineering_log.append(entry)
        logger.info(f"  [{status.upper()}] {operation} on '{column}': {details}")

    def one_hot_encode(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping one-hot encoding")
            return self.df
        if not (self.df[column].dtype == object or self.df[column].dtype.name == 'category'):
            logger.warning(f"  Column '{column}' is not categorical, skipping one-hot encoding")
            return self.df
        if self.df[column].isna().any():
            self.df[column] = self.df[column].fillna('MISSING')
        unique_count = self.df[column].nunique()
        max_categories = settings.ONE_HOT_MAX_CATEGORIES
        if unique_count > max_categories:
            logger.info(f"  Column '{column}' has {unique_count} categories (exceeds max {max_categories}), falling back to LabelEncoder")
            return self.label_encode(column)
        try:
            dummies = pd.get_dummies(self.df[column], prefix=column, prefix_sep='_', dtype=int)
            categories = self.df[column].unique().tolist()
            self.encoder_configs[column] = {'type': 'one_hot', 'categories': [str(c) for c in categories], 'new_columns': list(dummies.columns)}
            col_position = self.df.columns.get_loc(column)
            self.df = self.df.drop(columns=[column])
            for (i, dummy_col) in enumerate(dummies.columns):
                self.df.insert(col_position + i, dummy_col, dummies[dummy_col])
            self._log_operation(operation='one_hot_encoding', column=column, details=f'One-hot encoded {unique_count} categories into {len(dummies.columns)} binary columns', columns_added=len(dummies.columns), columns_removed=1)
        except Exception as e:
            self._log_operation(operation='one_hot_encoding', column=column, details=f'One-hot encoding failed: {str(e)}', status='error')
            logger.error(f"  One-hot encoding failed for '{column}': {str(e)}")
        return self.df

    def label_encode(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping label encoding")
            return self.df
        if not (self.df[column].dtype == object or self.df[column].dtype.name == 'category'):
            logger.warning(f"  Column '{column}' is not categorical, skipping label encoding")
            return self.df
        if self.df[column].isna().any():
            self.df[column] = self.df[column].fillna('MISSING')
        try:
            encoder = LabelEncoder()
            original_values = self.df[column].astype(str).values
            encoded_values = encoder.fit_transform(original_values)
            self.df[column] = encoded_values
            self.encoder_configs[column] = {'type': 'label', 'classes': encoder.classes_.tolist(), 'mapping': {str(cls): int(idx) for (idx, cls) in enumerate(encoder.classes_)}}
            unique_count = len(encoder.classes_)
            self._log_operation(operation='label_encoding', column=column, details=f'Label encoded {unique_count} categories to integers [0, {unique_count - 1}]', columns_added=0, columns_removed=0)
        except Exception as e:
            self._log_operation(operation='label_encoding', column=column, details=f'Label encoding failed: {str(e)}', status='error')
            logger.error(f"  Label encoding failed for '{column}': {str(e)}")
        return self.df

    def standard_scale(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping standard scaling")
            return self.df
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            logger.warning(f"  Column '{column}' is not numeric, skipping standard scaling")
            return self.df
        if self.df[column].std() == 0:
            logger.info(f"  Column '{column}' has zero variance, skipping standard scaling")
            return self.df
        try:
            scaler = StandardScaler()
            values = self.df[column].values.reshape(-1, 1)
            non_null_mask = ~np.isnan(values.ravel())
            if non_null_mask.sum() == 0:
                return self.df
            scaler.fit(values[non_null_mask].reshape(-1, 1))
            scaled_values = values.copy()
            scaled_values[non_null_mask] = scaler.transform(values[non_null_mask].reshape(-1, 1)).ravel()
            self.df[column] = scaled_values.ravel()
            self.scaler_configs[column] = {'type': 'standard', 'mean': float(scaler.mean_[0]), 'scale': float(scaler.scale_[0])}
            self._log_operation(operation='standard_scaling', column=column, details=f'StandardScaler applied (mean={scaler.mean_[0]:.4f}, scale={scaler.scale_[0]:.4f})')
        except Exception as e:
            self._log_operation(operation='standard_scaling', column=column, details=f'Standard scaling failed: {str(e)}', status='error')
            logger.error(f"  Standard scaling failed for '{column}': {str(e)}")
        return self.df

    def minmax_scale(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping minmax scaling")
            return self.df
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            logger.warning(f"  Column '{column}' is not numeric, skipping minmax scaling")
            return self.df
        col_min = self.df[column].min()
        col_max = self.df[column].max()
        if col_min == col_max:
            logger.info(f"  Column '{column}' has zero range, skipping minmax scaling")
            return self.df
        try:
            scaler = MinMaxScaler()
            values = self.df[column].values.reshape(-1, 1)
            non_null_mask = ~np.isnan(values.ravel())
            if non_null_mask.sum() == 0:
                return self.df
            scaler.fit(values[non_null_mask].reshape(-1, 1))
            scaled_values = values.copy()
            scaled_values[non_null_mask] = scaler.transform(values[non_null_mask].reshape(-1, 1)).ravel()
            self.df[column] = scaled_values.ravel()
            self.scaler_configs[column] = {'type': 'minmax', 'data_min': float(scaler.data_min_[0]), 'data_max': float(scaler.data_max_[0]), 'feature_range': (0, 1)}
            self._log_operation(operation='minmax_scaling', column=column, details=f'MinMaxScaler applied (min={scaler.data_min_[0]:.4f}, max={scaler.data_max_[0]:.4f})')
        except Exception as e:
            self._log_operation(operation='minmax_scaling', column=column, details=f'MinMax scaling failed: {str(e)}', status='error')
            logger.error(f"  MinMax scaling failed for '{column}': {str(e)}")
        return self.df

    def robust_scale(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping robust scaling")
            return self.df
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            logger.warning(f"  Column '{column}' is not numeric, skipping robust scaling")
            return self.df
        try:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            values = self.df[column].values.reshape(-1, 1)
            non_null_mask = ~np.isnan(values.ravel())
            if non_null_mask.sum() == 0:
                return self.df
            scaler.fit(values[non_null_mask].reshape(-1, 1))
            scaled_values = values.copy()
            scaled_values[non_null_mask] = scaler.transform(values[non_null_mask].reshape(-1, 1)).ravel()
            self.df[column] = scaled_values.ravel()
            self.scaler_configs[column] = {'type': 'robust', 'center': float(scaler.center_[0]), 'scale': float(scaler.scale_[0])}
            self._log_operation(operation='robust_scaling', column=column, details=f'RobustScaler applied (center={scaler.center_[0]:.4f}, scale={scaler.scale_[0]:.4f})')
        except Exception as e:
            self._log_operation(operation='robust_scaling', column=column, details=f'Robust scaling failed: {str(e)}', status='error')
            logger.error(f"  Robust scaling failed for '{column}': {str(e)}")
        return self.df

    def decompose_date(self, column: str, parts: Optional[List[str]]=None) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping date decomposition")
            return self.df
        if parts is None:
            parts = ['year', 'month', 'day', 'weekday']
        try:
            if not pd.api.types.is_datetime64_any_dtype(self.df[column]):
                self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
            if self.df[column].isna().all():
                self._log_operation(operation='date_decomposition', column=column, details='Date decomposition skipped: all values are NaT after conversion', status='warning')
                return self.df
            col_position = self.df.columns.get_loc(column)
            new_cols_added = 0
            part_extractors = {'year': lambda s: s.dt.year, 'month': lambda s: s.dt.month, 'day': lambda s: s.dt.day, 'weekday': lambda s: s.dt.weekday, 'hour': lambda s: s.dt.hour, 'minute': lambda s: s.dt.minute, 'quarter': lambda s: s.dt.quarter, 'day_of_year': lambda s: s.dt.dayofyear, 'week_of_year': lambda s: s.dt.isocalendar().week.astype(int)}
            for part in parts:
                part_lower = part.lower()
                if part_lower in part_extractors:
                    new_col_name = f'{column}_{part_lower}'
                    try:
                        self.df.insert(col_position + new_cols_added + 1, new_col_name, part_extractors[part_lower](self.df[column]))
                        new_cols_added += 1
                    except Exception as extract_err:
                        logger.warning(f"  Failed to extract {part_lower} from '{column}': {str(extract_err)}")
                else:
                    logger.warning(f"  Unknown date part '{part_lower}', skipping")
            if new_cols_added > 0:
                self.df = self.df.drop(columns=[column])
                self._log_operation(operation='date_decomposition', column=column, details=f"Decomposed into {new_cols_added} components: {', '.join(parts[:new_cols_added])}", columns_added=new_cols_added, columns_removed=1)
            else:
                self._log_operation(operation='date_decomposition', column=column, details='No date parts could be extracted', status='warning')
        except Exception as e:
            self._log_operation(operation='date_decomposition', column=column, details=f'Date decomposition failed: {str(e)}', status='error')
            logger.error(f"  Date decomposition failed for '{column}': {str(e)}")
        return self.df

    def select_features(self, columns_to_keep: Optional[List[str]]=None, columns_to_drop: Optional[List[str]]=None) -> pd.DataFrame:
        if columns_to_keep is not None:
            existing = [col for col in columns_to_keep if col in self.df.columns]
            dropped_count = len(self.df.columns) - len(existing)
            if existing:
                self.df = self.df[existing]
                self._log_operation(operation='feature_selection', column='MULTIPLE', details=f'Selected {len(existing)} features, dropped {dropped_count} features', columns_removed=dropped_count)
        elif columns_to_drop is not None:
            to_drop = [col for col in columns_to_drop if col in self.df.columns]
            if to_drop:
                self.df = self.df.drop(columns=to_drop)
                self._log_operation(operation='feature_selection', column=', '.join(to_drop), details=f"Dropped {len(to_drop)} features: {', '.join(to_drop)}", columns_removed=len(to_drop))
        return self.df

    def execute_plan(self, actions: List[PreprocessingAction]) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        logger.info('=' * 60)
        logger.info('STARTING FEATURE ENGINEERING PIPELINE')
        logger.info('=' * 60)
        logger.info(f'  Initial shape: {self.df.shape}')
        logger.info(f'  Total actions to process: {len(actions)}')
        logger.info('-' * 40)
        logger.info('Pass 1: Date Decomposition')
        logger.info('-' * 40)
        date_actions_executed = 0
        for action in actions:
            if action.feature_removal:
                continue
            col_name = action.column_name
            normalized_col = re.sub('[^a-zA-Z0-9_]', '', re.sub('[\\s\\-]+', '_', col_name.strip())).lower().strip('_')
            if normalized_col not in self.df.columns:
                continue
            if action.date_decomposition and len(action.date_decomposition) > 0:
                self.decompose_date(normalized_col, action.date_decomposition)
                date_actions_executed += 1
        if date_actions_executed == 0:
            logger.info('  No date decomposition actions found')
        if settings.ENABLE_DATE_DECOMPOSITION:
            datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
            for col in datetime_cols:
                if col not in self.encoder_configs:
                    logger.info(f'  Auto-decomposing detected datetime column: {col}')
                    self.decompose_date(col)
        logger.info('-' * 40)
        logger.info('Pass 2: Categorical Encoding')
        logger.info('-' * 40)
        encoding_actions_executed = 0
        encoded_columns: List[str] = []
        for action in actions:
            if action.feature_removal:
                continue
            col_name = action.column_name
            normalized_col = re.sub('[^a-zA-Z0-9_]', '', re.sub('[\\s\\-]+', '_', col_name.strip())).lower().strip('_')
            if normalized_col not in self.df.columns:
                continue
            if action.encoding_strategy and action.encoding_strategy not in ('none', 'None', None):
                strategy = action.encoding_strategy.lower()
                if strategy == 'one_hot':
                    self.one_hot_encode(normalized_col)
                    encoded_columns.append(normalized_col)
                    encoding_actions_executed += 1
                elif strategy in ('label', 'ordinal'):
                    self.label_encode(normalized_col)
                    encoded_columns.append(normalized_col)
                    encoding_actions_executed += 1
                else:
                    logger.warning(f"  Unknown encoding strategy '{strategy}' for '{normalized_col}'")
        remaining_categoricals = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in remaining_categoricals:
            if col not in encoded_columns:
                unique_count = self.df[col].nunique()
                if unique_count <= settings.ONE_HOT_MAX_CATEGORIES:
                    logger.info(f"  Auto-encoding remaining categorical column '{col}' with OneHot")
                    self.one_hot_encode(col)
                else:
                    logger.info(f"  Auto-encoding remaining categorical column '{col}' with Label")
                    self.label_encode(col)
                encoding_actions_executed += 1
        if encoding_actions_executed == 0:
            logger.info('  No categorical encoding actions needed')
        logger.info('-' * 40)
        logger.info('Pass 3: Numerical Scaling')
        logger.info('-' * 40)
        scaling_actions_executed = 0
        scaled_columns: List[str] = []
        for action in actions:
            if action.feature_removal:
                continue
            col_name = action.column_name
            normalized_col = re.sub('[^a-zA-Z0-9_]', '', re.sub('[\\s\\-]+', '_', col_name.strip())).lower().strip('_')
            if normalized_col not in self.df.columns:
                continue
            if action.scaling_strategy and action.scaling_strategy not in ('none', 'None', None):
                strategy = action.scaling_strategy.lower()
                if strategy == 'standard':
                    self.standard_scale(normalized_col)
                    scaled_columns.append(normalized_col)
                    scaling_actions_executed += 1
                elif strategy == 'minmax':
                    self.minmax_scale(normalized_col)
                    scaled_columns.append(normalized_col)
                    scaling_actions_executed += 1
                elif strategy == 'robust':
                    self.robust_scale(normalized_col)
                    scaled_columns.append(normalized_col)
                    scaling_actions_executed += 1
                else:
                    logger.warning(f"  Unknown scaling strategy '{strategy}' for '{normalized_col}'")
        if scaling_actions_executed == 0:
            logger.info('  No numerical scaling actions needed')
        logger.info('=' * 60)
        logger.info('FEATURE ENGINEERING COMPLETE')
        logger.info(f'  Original shape: {self.original_shape}')
        logger.info(f'  Processed shape: {self.df.shape}')
        logger.info(f'  Total operations: {len(self.engineering_log)}')
        logger.info(f'  Encoders created: {len(self.encoder_configs)}')
        logger.info(f'  Scalers created: {len(self.scaler_configs)}')
        logger.info('=' * 60)
        return (self.df, self.engineering_log, self.encoder_configs, self.scaler_configs)

async def feature_engineering_node(state: PreprocessingGraphState) -> PreprocessingGraphState:
    try:
        logger.info(f'[Node 4] Starting feature engineering for session {state.session_id}')
        state.add_log('Starting feature engineering...')
        if state.cleaned_dataframe is None:
            raise FeatureEngineeringException('Cleaned dataframe not found in state')
        if state.preprocessing_plan is None:
            raise FeatureEngineeringException('Preprocessing plan not found in state')
        logger.info('Reconstructing cleaned DataFrame from state...')
        df = pd.DataFrame(**state.cleaned_dataframe)
        state.add_log(f'Cleaned DataFrame reconstructed: {df.shape[0]} rows, {df.shape[1]} columns')
        executor = FeatureEngineeringExecutor(df)
        actions = state.preprocessing_plan.actions
        logger.info(f'Executing feature engineering with {len(actions)} actions...')
        state.add_log(f'Executing feature engineering ({len(actions)} actions)...')
        (processed_df, engineering_log, encoder_configs, scaler_configs) = executor.execute_plan(actions=actions)
        state.processed_dataframe = processed_df.to_dict(orient='split')
        state.feature_engineering_log = engineering_log
        state.encoder_configs = encoder_configs
        state.scaler_configs = scaler_configs
        success_count = sum((1 for entry in engineering_log if entry.get('status') == 'success'))
        warning_count = sum((1 for entry in engineering_log if entry.get('status') == 'warning'))
        error_count = sum((1 for entry in engineering_log if entry.get('status') == 'error'))
        summary = f'Feature engineering complete: {len(engineering_log)} operations ({success_count} success, {warning_count} warnings, {error_count} errors). Shape: {df.shape} -> {processed_df.shape}. Encoders: {len(encoder_configs)}, Scalers: {len(scaler_configs)}'
        logger.info(summary)
        state.add_log(summary)
        state.updated_at = datetime.now().isoformat()
        logger.info('[Node 4] Feature engineering completed successfully')
        return state
    except FeatureEngineeringException as e:
        logger.error(f'[Node 4] Feature engineering failed: {str(e)}')
        state.mark_failed(f'Feature engineering error: {str(e)}')
        state.add_log(f'ERROR: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'[Node 4] Unexpected error in feature engineering: {str(e)}')
        error_msg = f'Unexpected error during feature engineering: {str(e)}'
        state.mark_failed(error_msg)
        state.add_log(f'ERROR: {error_msg}')
        raise FeatureEngineeringException(error_msg) from e