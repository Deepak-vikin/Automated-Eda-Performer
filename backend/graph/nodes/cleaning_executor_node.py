import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from backend.config import settings
from backend.graph.state import PreprocessingGraphState
from backend.models.schemas import CleaningAction, PreprocessingAction
from backend.utils.exceptions import PreprocessingException
logger = logging.getLogger(__name__)

class CleaningExecutor:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_actions: List[CleaningAction] = []
        self.original_shape = df.shape

    def _record_action(self, column_name: str, action_type: str, action_details: str, rows_affected: int=0, status: str='success') -> None:
        action = CleaningAction(column_name=column_name, action_type=action_type, action_details=action_details, rows_affected=rows_affected, status=status)
        self.cleaning_actions.append(action)
        logger.info(f'  [{status.upper()}] {column_name}: {action_details} ({rows_affected} rows affected)')

    def normalize_column_names(self) -> pd.DataFrame:
        logger.info('Normalizing column names...')
        original_columns = list(self.df.columns)
        new_columns: List[str] = []
        for col in original_columns:
            normalized = str(col).strip()
            normalized = re.sub('[\\s\\-]+', '_', normalized)
            normalized = re.sub('[^a-zA-Z0-9_]', '', normalized)
            normalized = normalized.lower()
            normalized = normalized.strip('_')
            normalized = re.sub('_+', '_', normalized)
            if not normalized:
                normalized = f'column_{len(new_columns)}'
            new_columns.append(normalized)
        seen: Dict[str, int] = {}
        final_columns: List[str] = []
        for col in new_columns:
            if col in seen:
                seen[col] += 1
                final_columns.append(f'{col}_{seen[col]}')
            else:
                seen[col] = 0
                final_columns.append(col)
        renamed_count = sum((1 for (old, new) in zip(original_columns, final_columns) if old != new))
        if renamed_count > 0:
            self.df.columns = final_columns
            self._record_action(column_name='ALL_COLUMNS', action_type='column_name_normalization', action_details=f'Normalized {renamed_count} column names to snake_case', rows_affected=0, status='success')
            for (old, new) in zip(original_columns, final_columns):
                if old != new:
                    logger.debug(f"  Renamed: '{old}' -> '{new}'")
        else:
            logger.info('  No column name changes needed')
        return self.df

    def normalize_nulls(self) -> pd.DataFrame:
        logger.info('Normalizing null representations...')
        null_representations = ['', 'null', 'NULL', 'None', 'none', 'N/A', 'n/a', 'NA', 'na', '-', '--', '?', 'missing', 'MISSING', 'undefined', 'NaN', 'nan', 'NAN', 'nil', 'NIL', '#N/A', '#NA', 'N/a']
        total_converted = 0
        for col in self.df.columns:
            if self.df[col].dtype == object:
                mask = self.df[col].isin(null_representations)
                whitespace_mask = self.df[col].astype(str).str.strip().isin(null_representations)
                combined_mask = mask | whitespace_mask
                count = int(combined_mask.sum())
                if count > 0:
                    self.df.loc[combined_mask, col] = np.nan
                    total_converted += count
                    self._record_action(column_name=col, action_type='null_normalization', action_details=f'Converted {count} null-like values to NaN', rows_affected=count, status='success')
        if total_converted == 0:
            logger.info('  No null-like values found to normalize')
        return self.df

    def cleanup_whitespace(self) -> pd.DataFrame:
        logger.info('Cleaning up whitespace in string columns...')
        for col in self.df.select_dtypes(include=['object']).columns:
            original_values = self.df[col].copy()
            self.df[col] = self.df[col].astype(str).str.strip()
            self.df.loc[self.df[col] == 'nan', col] = np.nan
            changed_mask = (original_values != self.df[col]) & original_values.notna()
            changed_count = int(changed_mask.sum())
            if changed_count > 0:
                self._record_action(column_name=col, action_type='whitespace_cleanup', action_details=f'Stripped whitespace from {changed_count} values', rows_affected=changed_count, status='success')
        return self.df

    def remove_duplicates(self) -> pd.DataFrame:
        logger.info('Checking for duplicate rows...')
        duplicate_count = int(self.df.duplicated().sum())
        if duplicate_count > 0:
            original_rows = len(self.df)
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            self._record_action(column_name='ALL_COLUMNS', action_type='duplicate_removal', action_details=f'Removed {duplicate_count} duplicate rows ({original_rows} -> {len(self.df)} rows)', rows_affected=duplicate_count, status='success')
        else:
            logger.info('  No duplicate rows found')
        return self.df

    def impute_mean(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping mean imputation")
            return self.df
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            logger.warning(f"  Column '{column}' is not numeric, skipping mean imputation")
            return self.df
        missing_count = int(self.df[column].isna().sum())
        if missing_count > 0:
            mean_value = self.df[column].mean()
            self.df[column] = self.df[column].fillna(mean_value)
            self._record_action(column_name=column, action_type='imputation', action_details=f'Mean imputation: filled {missing_count} missing values with {mean_value:.4f}', rows_affected=missing_count, status='success')
        else:
            logger.debug(f"  No missing values in '{column}' for mean imputation")
        return self.df

    def impute_median(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping median imputation")
            return self.df
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            logger.warning(f"  Column '{column}' is not numeric, skipping median imputation")
            return self.df
        missing_count = int(self.df[column].isna().sum())
        if missing_count > 0:
            median_value = self.df[column].median()
            self.df[column] = self.df[column].fillna(median_value)
            self._record_action(column_name=column, action_type='imputation', action_details=f'Median imputation: filled {missing_count} missing values with {median_value:.4f}', rows_affected=missing_count, status='success')
        else:
            logger.debug(f"  No missing values in '{column}' for median imputation")
        return self.df

    def impute_mode(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping mode imputation")
            return self.df
        missing_count = int(self.df[column].isna().sum())
        if missing_count > 0:
            mode_values = self.df[column].mode()
            if len(mode_values) > 0:
                mode_value = mode_values.iloc[0]
                self.df[column] = self.df[column].fillna(mode_value)
                self._record_action(column_name=column, action_type='imputation', action_details=f"Mode imputation: filled {missing_count} missing values with '{mode_value}'", rows_affected=missing_count, status='success')
            else:
                self._record_action(column_name=column, action_type='imputation', action_details='Mode imputation skipped: no mode value found', rows_affected=0, status='warning')
        else:
            logger.debug(f"  No missing values in '{column}' for mode imputation")
        return self.df

    def drop_missing(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping drop missing")
            return self.df
        missing_count = int(self.df[column].isna().sum())
        if missing_count > 0:
            original_rows = len(self.df)
            self.df = self.df.dropna(subset=[column]).reset_index(drop=True)
            self._record_action(column_name=column, action_type='imputation', action_details=f'Dropped {missing_count} rows with missing values ({original_rows} -> {len(self.df)} rows)', rows_affected=missing_count, status='success')
        return self.df

    def handle_outliers_iqr(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping IQR outlier handling")
            return self.df
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            logger.warning(f"  Column '{column}' is not numeric, skipping IQR outlier handling")
            return self.df
        non_null_data = self.df[column].dropna()
        if len(non_null_data) == 0:
            return self.df
        q1 = non_null_data.quantile(0.25)
        q3 = non_null_data.quantile(0.75)
        iqr = q3 - q1
        multiplier = settings.IQR_MULTIPLIER
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
            self._record_action(column_name=column, action_type='outlier_handling', action_details=f'IQR outlier clipping: {outlier_count} outliers clipped [bounds: {lower_bound:.4f}, {upper_bound:.4f}]', rows_affected=outlier_count, status='success')
        else:
            logger.debug(f"  No outliers found in '{column}' using IQR method")
        return self.df

    def handle_outliers_zscore(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping Z-score outlier handling")
            return self.df
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            logger.warning(f"  Column '{column}' is not numeric, skipping Z-score outlier handling")
            return self.df
        non_null_data = self.df[column].dropna()
        if len(non_null_data) == 0:
            return self.df
        mean = non_null_data.mean()
        std = non_null_data.std()
        if std == 0:
            logger.debug(f"  Column '{column}' has zero std deviation, skipping Z-score")
            return self.df
        threshold = settings.ZSCORE_THRESHOLD
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
            self._record_action(column_name=column, action_type='outlier_handling', action_details=f'Z-score outlier clipping: {outlier_count} outliers clipped [bounds: {lower_bound:.4f}, {upper_bound:.4f}]', rows_affected=outlier_count, status='success')
        else:
            logger.debug(f"  No outliers found in '{column}' using Z-score method")
        return self.df

    def drop_outliers(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping outlier drop")
            return self.df
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return self.df
        non_null_data = self.df[column].dropna()
        if len(non_null_data) == 0:
            return self.df
        q1 = non_null_data.quantile(0.25)
        q3 = non_null_data.quantile(0.75)
        iqr = q3 - q1
        multiplier = settings.IQR_MULTIPLIER
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            original_rows = len(self.df)
            self.df = self.df[~outlier_mask].reset_index(drop=True)
            self._record_action(column_name=column, action_type='outlier_handling', action_details=f'Dropped {outlier_count} outlier rows ({original_rows} -> {len(self.df)} rows)', rows_affected=outlier_count, status='success')
        return self.df

    def convert_datatype(self, column: str, target_type: str) -> pd.DataFrame:
        if column not in self.df.columns:
            logger.warning(f"  Column '{column}' not found, skipping datatype conversion")
            return self.df
        if target_type in ('none', 'None', None, ''):
            return self.df
        original_dtype = str(self.df[column].dtype)
        try:
            type_mapping = {'int': 'int64', 'int32': 'int32', 'int64': 'int64', 'float': 'float64', 'float32': 'float32', 'float64': 'float64', 'str': 'object', 'string': 'object', 'object': 'object', 'bool': 'bool', 'boolean': 'bool', 'category': 'category', 'datetime': 'datetime64[ns]'}
            target = type_mapping.get(target_type.lower(), target_type)
            if target == 'datetime64[ns]':
                self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
            elif target in ('int64', 'int32'):
                if self.df[column].isna().any():
                    self.df[column] = self.df[column].fillna(0)
                self.df[column] = self.df[column].astype(target)
            elif target == 'bool':
                self.df[column] = self.df[column].astype(bool)
            else:
                self.df[column] = self.df[column].astype(target)
            new_dtype = str(self.df[column].dtype)
            if original_dtype != new_dtype:
                self._record_action(column_name=column, action_type='datatype_conversion', action_details=f'Converted from {original_dtype} to {new_dtype}', rows_affected=len(self.df), status='success')
        except Exception as e:
            self._record_action(column_name=column, action_type='datatype_conversion', action_details=f'Failed to convert to {target_type}: {str(e)}', rows_affected=0, status='error')
            logger.error(f"  Datatype conversion failed for '{column}': {str(e)}")
        return self.df

    def remove_features(self, columns: List[str]) -> pd.DataFrame:
        columns_to_drop = [col for col in columns if col in self.df.columns]
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)
            self._record_action(column_name=', '.join(columns_to_drop), action_type='feature_removal', action_details=f"Removed {len(columns_to_drop)} features: {', '.join(columns_to_drop)}", rows_affected=0, status='success')
        not_found = [col for col in columns if col not in self.df.columns and col not in columns_to_drop]
        if not_found:
            logger.warning(f'  Columns not found for removal: {not_found}')
        return self.df

    def drop_high_nan_columns(self) -> pd.DataFrame:
        logger.info('Checking for columns exceeding NaN threshold...')
        threshold = settings.NAN_THRESHOLD_PERCENTAGE
        columns_to_drop: List[str] = []
        for col in self.df.columns:
            nan_pct = self.df[col].isna().sum() / len(self.df)
            if nan_pct > threshold:
                columns_to_drop.append(col)
                logger.debug(f"  Column '{col}' has {nan_pct * 100:.1f}% NaN (threshold: {threshold * 100:.1f}%)")
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)
            self._record_action(column_name=', '.join(columns_to_drop), action_type='high_nan_removal', action_details=f"Dropped {len(columns_to_drop)} columns exceeding {threshold * 100:.0f}% NaN threshold: {', '.join(columns_to_drop)}", rows_affected=0, status='success')
        else:
            logger.info(f'  No columns exceed the {threshold * 100:.0f}% NaN threshold')
        return self.df

    def execute_plan(self, actions: List[PreprocessingAction], features_to_remove: List[str]) -> Tuple[pd.DataFrame, List[CleaningAction]]:
        logger.info('=' * 60)
        logger.info('STARTING DATA CLEANING PIPELINE')
        logger.info('=' * 60)
        logger.info(f'  Initial shape: {self.df.shape}')
        self.normalize_column_names()
        self.normalize_nulls()
        self.cleanup_whitespace()
        self.drop_high_nan_columns()
        if features_to_remove:
            normalized_removals = [re.sub('[^a-zA-Z0-9_]', '', re.sub('[\\s\\-]+', '_', col.strip())).lower().strip('_') for col in features_to_remove]
            self.remove_features(normalized_removals)
        for action in actions:
            col_name = action.column_name
            normalized_col = re.sub('[^a-zA-Z0-9_]', '', re.sub('[\\s\\-]+', '_', col_name.strip())).lower().strip('_')
            if normalized_col not in self.df.columns:
                logger.debug(f"  Skipping column '{col_name}' (not in DataFrame after normalization)")
                continue
            if action.feature_removal:
                continue
            if action.missing_strategy and action.missing_strategy not in ('none', 'None', None):
                strategy = action.missing_strategy.lower()
                if strategy == 'mean':
                    self.impute_mean(normalized_col)
                elif strategy == 'median':
                    self.impute_median(normalized_col)
                elif strategy == 'mode':
                    self.impute_mode(normalized_col)
                elif strategy == 'drop':
                    self.drop_missing(normalized_col)
                else:
                    logger.warning(f"  Unknown imputation strategy '{strategy}' for '{normalized_col}'")
            if action.outlier_strategy and action.outlier_strategy not in ('none', 'None', None):
                strategy = action.outlier_strategy.lower()
                if strategy == 'iqr':
                    self.handle_outliers_iqr(normalized_col)
                elif strategy == 'zscore':
                    self.handle_outliers_zscore(normalized_col)
                elif strategy == 'drop':
                    self.drop_outliers(normalized_col)
                else:
                    logger.warning(f"  Unknown outlier strategy '{strategy}' for '{normalized_col}'")
            if action.datatype_conversion and action.datatype_conversion not in ('none', 'None', None):
                self.convert_datatype(normalized_col, action.datatype_conversion)
        self.remove_duplicates()
        logger.info('=' * 60)
        logger.info('DATA CLEANING COMPLETE')
        logger.info(f'  Original shape: {self.original_shape}')
        logger.info(f'  Cleaned shape:  {self.df.shape}')
        logger.info(f'  Total cleaning actions: {len(self.cleaning_actions)}')
        logger.info('=' * 60)
        return (self.df, self.cleaning_actions)

async def cleaning_executor_node(state: PreprocessingGraphState) -> PreprocessingGraphState:
    try:
        logger.info(f'[Node 3] Starting cleaning execution for session {state.session_id}')
        state.add_log('Starting data cleaning...')
        if state.original_dataframe is None:
            raise PreprocessingException('Original dataframe not found in state')
        if state.preprocessing_plan is None:
            raise PreprocessingException('Preprocessing plan not found in state')
        logger.info('Reconstructing DataFrame from state...')
        df = pd.DataFrame(**state.original_dataframe)
        state.add_log(f'DataFrame reconstructed: {df.shape[0]} rows, {df.shape[1]} columns')
        executor = CleaningExecutor(df)
        actions = state.preprocessing_plan.actions
        features_to_remove = state.preprocessing_plan.features_to_remove
        logger.info(f'Executing cleaning plan with {len(actions)} actions...')
        state.add_log(f'Executing {len(actions)} cleaning actions...')
        (cleaned_df, cleaning_actions) = executor.execute_plan(actions=actions, features_to_remove=features_to_remove)
        state.cleaned_dataframe = cleaned_df.to_dict(orient='split')
        state.cleaning_actions = cleaning_actions
        success_count = sum((1 for a in cleaning_actions if a.status == 'success'))
        warning_count = sum((1 for a in cleaning_actions if a.status == 'warning'))
        error_count = sum((1 for a in cleaning_actions if a.status == 'error'))
        summary = f'Cleaning complete: {len(cleaning_actions)} total actions ({success_count} success, {warning_count} warnings, {error_count} errors). Shape: {df.shape} -> {cleaned_df.shape}'
        logger.info(summary)
        state.add_log(summary)
        state.updated_at = datetime.now().isoformat()
        logger.info('[Node 3] Cleaning execution completed successfully')
        return state
    except PreprocessingException as e:
        logger.error(f'[Node 3] Cleaning execution failed: {str(e)}')
        state.mark_failed(f'Cleaning execution error: {str(e)}')
        state.add_log(f'ERROR: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'[Node 3] Unexpected error in cleaning execution: {str(e)}')
        error_msg = f'Unexpected error during cleaning: {str(e)}'
        state.mark_failed(error_msg)
        state.add_log(f'ERROR: {error_msg}')
        raise PreprocessingException(error_msg) from e