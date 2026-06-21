import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from backend.config import settings
from backend.graph.state import PreprocessingGraphState
from backend.models.schemas import ReadinessReport
from backend.utils.exceptions import ReadinessValidationException
logger = logging.getLogger(__name__)

class ReadinessValidator:
    WEIGHT_MISSING_VALUES = 25
    WEIGHT_DUPLICATES = 10
    WEIGHT_ENCODING = 20
    WEIGHT_SCALING = 15
    WEIGHT_DATATYPES = 15
    WEIGHT_COMPLETENESS = 15

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.issues: List[str] = []
        self.recommendations: List[str] = []
        self.checks: Dict[str, bool] = {}

    def check_missing_values(self) -> Tuple[float, int]:
        logger.info('  Checking missing values...')
        total_missing = int(self.df.isna().sum().sum())
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_pct = total_missing / total_cells * 100 if total_cells > 0 else 0
        if total_missing == 0:
            score = self.WEIGHT_MISSING_VALUES
            self.checks['no_missing_values'] = True
            logger.info('    ✅ No missing values')
        elif missing_pct < 1:
            score = self.WEIGHT_MISSING_VALUES * 0.8
            self.checks['no_missing_values'] = False
            self.issues.append(f'Found {total_missing} missing values ({missing_pct:.2f}% of data)')
            self.recommendations.append('Apply imputation to handle remaining missing values')
            logger.info(f'    ⚠️ {total_missing} missing values ({missing_pct:.2f}%)')
        elif missing_pct < 5:
            score = self.WEIGHT_MISSING_VALUES * 0.5
            self.checks['no_missing_values'] = False
            self.issues.append(f'Found {total_missing} missing values ({missing_pct:.2f}% of data)')
            self.recommendations.append('Significant missing values remain. Consider additional imputation or row removal.')
            logger.info(f'    ❌ {total_missing} missing values ({missing_pct:.2f}%)')
        else:
            score = self.WEIGHT_MISSING_VALUES * 0.1
            self.checks['no_missing_values'] = False
            self.issues.append(f'High number of missing values: {total_missing} ({missing_pct:.2f}% of data)')
            self.recommendations.append('Critical: Too many missing values for reliable ML. Consider dropping features or re-imputing.')
            logger.info(f'    ❌ {total_missing} missing values ({missing_pct:.2f}%) — critical')
        return (score, total_missing)

    def check_duplicates(self) -> Tuple[float, int]:
        logger.info('  Checking duplicates...')
        duplicate_count = int(self.df.duplicated().sum())
        dup_pct = duplicate_count / len(self.df) * 100 if len(self.df) > 0 else 0
        if duplicate_count == 0:
            score = self.WEIGHT_DUPLICATES
            self.checks['no_duplicates'] = True
            logger.info('    ✅ No duplicate rows')
        elif dup_pct < 5:
            score = self.WEIGHT_DUPLICATES * 0.7
            self.checks['no_duplicates'] = False
            self.issues.append(f'Found {duplicate_count} duplicate rows ({dup_pct:.2f}%)')
            self.recommendations.append('Consider removing remaining duplicate rows')
            logger.info(f'    ⚠️ {duplicate_count} duplicates ({dup_pct:.2f}%)')
        else:
            score = self.WEIGHT_DUPLICATES * 0.3
            self.checks['no_duplicates'] = False
            self.issues.append(f'High number of duplicates: {duplicate_count} ({dup_pct:.2f}%)')
            self.recommendations.append('Significant duplicate rows detected. Remove duplicates before training.')
            logger.info(f'    ❌ {duplicate_count} duplicates ({dup_pct:.2f}%) — high')
        return (score, duplicate_count)

    def check_categorical_encoding(self) -> Tuple[float, bool]:
        logger.info('  Checking categorical encoding...')
        object_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(object_cols) == 0:
            score = self.WEIGHT_ENCODING
            self.checks['all_categoricals_encoded'] = True
            logger.info('    ✅ All categorical features are encoded')
            return (score, True)
        else:
            total_cols = len(self.df.columns)
            remaining_pct = len(object_cols) / total_cols if total_cols > 0 else 1
            score = self.WEIGHT_ENCODING * max(0, 1 - remaining_pct * 2)
            self.checks['all_categoricals_encoded'] = False
            self.issues.append(f"{len(object_cols)} categorical columns remain unencoded: {', '.join(object_cols[:5])}{('...' if len(object_cols) > 5 else '')}")
            self.recommendations.append(f"Encode remaining categorical columns: {', '.join(object_cols[:5])}")
            logger.info(f'    ❌ {len(object_cols)} unencoded categorical columns')
            return (score, False)

    def check_scaling(self) -> Tuple[float, bool]:
        logger.info('  Checking numerical scaling...')
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            self.checks['numerical_scaled'] = True
            logger.info('    ✅ No numeric columns to scale')
            return (self.WEIGHT_SCALING, True)
        scaled_count = 0
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            col_min = col_data.min()
            col_max = col_data.max()
            col_mean = col_data.mean()
            col_std = col_data.std()
            is_standard_scaled = abs(col_mean) < 0.5 and 0.5 < col_std < 2.0
            is_minmax_scaled = -0.1 <= col_min <= 0.1 and 0.9 <= col_max <= 1.1
            is_reasonable_range = col_max - col_min < 100
            if is_standard_scaled or is_minmax_scaled or is_reasonable_range:
                scaled_count += 1
        scaling_ratio = scaled_count / len(numeric_cols) if numeric_cols else 1
        score = self.WEIGHT_SCALING * scaling_ratio
        if scaling_ratio >= 0.9:
            self.checks['numerical_scaled'] = True
            logger.info(f'    ✅ {scaled_count}/{len(numeric_cols)} columns appear scaled')
        elif scaling_ratio >= 0.5:
            self.checks['numerical_scaled'] = False
            self.issues.append(f'Only {scaled_count}/{len(numeric_cols)} numerical columns appear scaled')
            self.recommendations.append('Apply StandardScaler or MinMaxScaler to unscaled numerical features')
            logger.info(f'    ⚠️ {scaled_count}/{len(numeric_cols)} columns scaled')
        else:
            self.checks['numerical_scaled'] = False
            self.issues.append(f'Most numerical columns ({len(numeric_cols) - scaled_count}/{len(numeric_cols)}) appear unscaled')
            self.recommendations.append('Scaling is recommended for most ML algorithms. Apply StandardScaler to all numerical features.')
            logger.info(f'    ❌ {scaled_count}/{len(numeric_cols)} columns scaled — insufficient')
        return (score, scaling_ratio >= 0.9)

    def check_datatypes(self) -> Tuple[float, bool]:
        logger.info('  Checking datatypes...')
        invalid_cols: List[str] = []
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                continue
            elif pd.api.types.is_bool_dtype(dtype):
                continue
            else:
                invalid_cols.append(f'{col} ({dtype})')
        if len(invalid_cols) == 0:
            score = self.WEIGHT_DATATYPES
            self.checks['all_datatypes_valid'] = True
            logger.info('    ✅ All datatypes are ML-compatible')
        else:
            valid_ratio = (len(self.df.columns) - len(invalid_cols)) / len(self.df.columns)
            score = self.WEIGHT_DATATYPES * valid_ratio
            self.checks['all_datatypes_valid'] = False
            self.issues.append(f"{len(invalid_cols)} columns have non-numeric datatypes: {', '.join(invalid_cols[:5])}{('...' if len(invalid_cols) > 5 else '')}")
            self.recommendations.append('Convert or encode remaining non-numeric columns')
            logger.info(f'    ❌ {len(invalid_cols)} columns with invalid datatypes')
        return (score, len(invalid_cols) == 0)

    def check_feature_completeness(self) -> Tuple[float, bool]:
        logger.info('  Checking feature completeness...')
        n_rows = len(self.df)
        n_cols = len(self.df.columns)
        if n_rows == 0 or n_cols == 0:
            self.checks['feature_completeness'] = False
            self.issues.append('Dataset is empty')
            return (0, False)
        ratio = n_rows / n_cols if n_cols > 0 else 0
        ratio_ok = ratio >= 10
        completeness_per_col = self.df.notna().mean()
        low_completeness_cols = completeness_per_col[completeness_per_col < 0.5].index.tolist()
        completeness_ok = len(low_completeness_cols) == 0
        numeric_df = self.df.select_dtypes(include=[np.number])
        constant_cols = []
        if len(numeric_df.columns) > 0:
            variances = numeric_df.var()
            constant_cols = variances[variances == 0].index.tolist()
        constant_ok = len(constant_cols) == 0
        all_ok = ratio_ok and completeness_ok and constant_ok
        if all_ok:
            score = self.WEIGHT_COMPLETENESS
            self.checks['feature_completeness'] = True
            logger.info(f'    ✅ Feature completeness OK (ratio: {ratio:.1f}:1)')
        else:
            deductions = 0
            if not ratio_ok:
                self.issues.append(f'Low samples-to-features ratio: {ratio:.1f}:1 (recommended ≥10:1)')
                self.recommendations.append('Consider feature selection to reduce dimensionality')
                deductions += 0.3
                logger.info(f'    ⚠️ Low ratio: {ratio:.1f}:1')
            if not completeness_ok:
                self.issues.append(f"{len(low_completeness_cols)} columns have <50% completeness: {', '.join(low_completeness_cols[:3])}")
                self.recommendations.append('Drop or impute columns with very low completeness')
                deductions += 0.3
                logger.info(f'    ⚠️ {len(low_completeness_cols)} low-completeness columns')
            if not constant_ok:
                self.issues.append(f"{len(constant_cols)} constant (zero-variance) columns: {', '.join(constant_cols[:3])}")
                self.recommendations.append('Remove constant columns as they provide no information')
                deductions += 0.2
                logger.info(f'    ⚠️ {len(constant_cols)} constant columns')
            score = self.WEIGHT_COMPLETENESS * max(0, 1 - deductions)
            self.checks['feature_completeness'] = False
        return (score, all_ok)

    def validate(self) -> ReadinessReport:
        logger.info('=' * 60)
        logger.info('STARTING ML READINESS VALIDATION')
        logger.info('=' * 60)
        logger.info(f'  Dataset shape: {self.df.shape}')
        (missing_score, remaining_missing) = self.check_missing_values()
        (duplicate_score, remaining_duplicates) = self.check_duplicates()
        (encoding_score, all_encoded) = self.check_categorical_encoding()
        (scaling_score, all_scaled) = self.check_scaling()
        (datatype_score, all_valid) = self.check_datatypes()
        (completeness_score, is_complete) = self.check_feature_completeness()
        total_score = int(round(missing_score + duplicate_score + encoding_score + scaling_score + datatype_score + completeness_score))
        total_score = max(0, min(100, total_score))
        ready_for_training = total_score >= settings.MIN_READINESS_SCORE
        report = ReadinessReport(validation_timestamp=datetime.now().isoformat(), remaining_missing_values=remaining_missing, remaining_duplicates=remaining_duplicates, categorical_encoded=all_encoded, numerical_scaled=all_scaled, all_datatypes_valid=all_valid, data_quality_checks=self.checks, readiness_score=total_score, ready_for_training=ready_for_training, issues=self.issues, recommendations=self.recommendations)
        logger.info('-' * 40)
        logger.info('SCORING BREAKDOWN:')
        logger.info(f'  Missing Values:   {missing_score:.1f} / {self.WEIGHT_MISSING_VALUES}')
        logger.info(f'  Duplicates:       {duplicate_score:.1f} / {self.WEIGHT_DUPLICATES}')
        logger.info(f'  Encoding:         {encoding_score:.1f} / {self.WEIGHT_ENCODING}')
        logger.info(f'  Scaling:          {scaling_score:.1f} / {self.WEIGHT_SCALING}')
        logger.info(f'  Datatypes:        {datatype_score:.1f} / {self.WEIGHT_DATATYPES}')
        logger.info(f'  Completeness:     {completeness_score:.1f} / {self.WEIGHT_COMPLETENESS}')
        logger.info('-' * 40)
        logger.info(f'  TOTAL SCORE:      {total_score} / 100')
        logger.info(f"  READY FOR ML:     {('✅ YES' if ready_for_training else '❌ NO')}")
        logger.info(f'  Issues:           {len(self.issues)}')
        logger.info(f'  Recommendations:  {len(self.recommendations)}')
        logger.info('=' * 60)
        return report

async def readiness_validator_node(state: PreprocessingGraphState) -> PreprocessingGraphState:
    try:
        logger.info(f'[Node 6] Starting ML readiness validation for session {state.session_id}')
        state.add_log('Starting ML readiness validation...')
        if state.processed_dataframe is None:
            raise ReadinessValidationException('Processed dataframe not found in state')
        logger.info('Reconstructing processed DataFrame for validation...')
        df = pd.DataFrame(**state.processed_dataframe)
        state.add_log(f'DataFrame for validation: {df.shape[0]} rows, {df.shape[1]} columns')
        validator = ReadinessValidator(df)
        logger.info('Running validation checks...')
        state.add_log('Running ML readiness checks...')
        report = validator.validate()
        state.readiness_report = report
        summary = f"ML Readiness Score: {report.readiness_score}/100 — {('Ready for training ✅' if report.ready_for_training else 'Not ready ❌')}. Issues: {len(report.issues)}, Recommendations: {len(report.recommendations)}"
        logger.info(summary)
        state.add_log(summary)
        if report.issues:
            for issue in report.issues:
                state.add_log(f'  Issue: {issue}')
        if report.recommendations:
            for rec in report.recommendations:
                state.add_log(f'  Recommendation: {rec}')
        state.updated_at = datetime.now().isoformat()
        logger.info('[Node 6] ML readiness validation completed successfully')
        return state
    except ReadinessValidationException as e:
        logger.error(f'[Node 6] Readiness validation failed: {str(e)}')
        state.mark_failed(f'Readiness validation error: {str(e)}')
        state.add_log(f'ERROR: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'[Node 6] Unexpected error in readiness validation: {str(e)}')
        error_msg = f'Unexpected error during readiness validation: {str(e)}'
        state.mark_failed(error_msg)
        state.add_log(f'ERROR: {error_msg}')
        raise ReadinessValidationException(error_msg) from e