import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from backend.config import settings
from backend.graph.state import PreprocessingGraphState
from backend.utils.exceptions import OutputGenerationException
logger = logging.getLogger(__name__)

class OutputGenerator:

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_files: List[str] = []

    def _save_file(self, filepath: Path, description: str) -> str:
        abs_path = str(filepath.resolve())
        self.generated_files.append(abs_path)
        logger.info(f'  Generated: {description} -> {abs_path}')
        return abs_path

    def export_cleaned_dataset(self, state: PreprocessingGraphState) -> str:
        logger.info('Exporting cleaned dataset...')
        if state.cleaned_dataframe is None:
            raise OutputGenerationException('Cleaned dataframe not available for export')
        try:
            df = pd.DataFrame(**state.cleaned_dataframe)
            filepath = self.output_dir / 'cleaned_dataset.csv'
            df.to_csv(filepath, index=False, encoding='utf-8')
            path = self._save_file(filepath, f'Cleaned dataset ({df.shape[0]} rows, {df.shape[1]} cols)')
            return path
        except Exception as e:
            raise OutputGenerationException(f'Failed to export cleaned dataset: {str(e)}') from e

    def export_processed_dataset(self, state: PreprocessingGraphState) -> str:
        logger.info('Exporting processed dataset...')
        if state.processed_dataframe is None:
            raise OutputGenerationException('Processed dataframe not available for export')
        try:
            df = pd.DataFrame(**state.processed_dataframe)
            filepath = self.output_dir / 'processed_dataset.csv'
            df.to_csv(filepath, index=False, encoding='utf-8')
            path = self._save_file(filepath, f'Processed dataset ({df.shape[0]} rows, {df.shape[1]} cols)')
            return path
        except Exception as e:
            raise OutputGenerationException(f'Failed to export processed dataset: {str(e)}') from e

    def generate_preprocessing_report(self, state: PreprocessingGraphState) -> str:
        logger.info('Generating preprocessing report...')
        try:
            report: Dict[str, Any] = {'report_metadata': {'generated_at': datetime.now().isoformat(), 'session_id': state.session_id, 'version': settings.APP_VERSION}, 'dataset_info': {'original_path': state.dataset_path, 'file_type': state.file_type}}
            if state.dataset_profile:
                report['dataset_profile'] = {'filename': state.dataset_profile.filename, 'row_count': state.dataset_profile.row_count, 'column_count': state.dataset_profile.column_count, 'total_missing_values': state.dataset_profile.total_missing_values, 'total_duplicate_rows': state.dataset_profile.total_duplicate_rows, 'memory_usage_mb': state.dataset_profile.memory_usage_mb, 'numerical_columns': state.dataset_profile.numerical_columns, 'categorical_columns': state.dataset_profile.categorical_columns, 'datetime_columns': state.dataset_profile.datetime_columns, 'columns': [{'name': col.name, 'data_type': col.data_type, 'missing_count': col.missing_count, 'missing_percentage': col.missing_percentage, 'unique_count': col.unique_count} for col in state.dataset_profile.columns]}
            if state.preprocessing_plan:
                report['preprocessing_plan'] = {'plan_timestamp': state.preprocessing_plan.plan_timestamp, 'total_columns': state.preprocessing_plan.total_columns, 'features_to_remove': state.preprocessing_plan.features_to_remove, 'target_column': state.preprocessing_plan.target_column, 'summary': state.preprocessing_plan.summary, 'actions': [{'column_name': action.column_name, 'missing_strategy': action.missing_strategy, 'outlier_strategy': action.outlier_strategy, 'encoding_strategy': action.encoding_strategy, 'scaling_strategy': action.scaling_strategy, 'datatype_conversion': action.datatype_conversion, 'feature_removal': action.feature_removal, 'date_decomposition': action.date_decomposition, 'reasoning': action.reasoning} for action in state.preprocessing_plan.actions]}
            if state.cleaning_actions:
                report['cleaning_actions'] = [{'column_name': action.column_name, 'action_type': action.action_type, 'action_details': action.action_details, 'rows_affected': action.rows_affected, 'status': action.status} for action in state.cleaning_actions]
            if state.feature_engineering_log:
                report['feature_engineering'] = {'operations': state.feature_engineering_log, 'encoder_configs': state.encoder_configs, 'scaler_configs': state.scaler_configs}
            original_shape = None
            cleaned_shape = None
            processed_shape = None
            if state.original_dataframe:
                try:
                    orig_df = pd.DataFrame(**state.original_dataframe)
                    original_shape = list(orig_df.shape)
                except Exception:
                    pass
            if state.cleaned_dataframe:
                try:
                    clean_df = pd.DataFrame(**state.cleaned_dataframe)
                    cleaned_shape = list(clean_df.shape)
                except Exception:
                    pass
            if state.processed_dataframe:
                try:
                    proc_df = pd.DataFrame(**state.processed_dataframe)
                    processed_shape = list(proc_df.shape)
                except Exception:
                    pass
            report['shape_transformations'] = {'original': original_shape, 'after_cleaning': cleaned_shape, 'after_feature_engineering': processed_shape}
            report['execution_logs'] = state.execution_logs
            filepath = self.output_dir / 'preprocessing_report.json'
            filepath.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding='utf-8')
            path = self._save_file(filepath, 'Preprocessing report')
            return path
        except OutputGenerationException:
            raise
        except Exception as e:
            raise OutputGenerationException(f'Failed to generate preprocessing report: {str(e)}') from e

    def generate_readiness_report(self, state: PreprocessingGraphState) -> str:
        logger.info('Generating readiness report...')
        if state.readiness_report is None:
            raise OutputGenerationException('Readiness report not available for export')
        try:
            report = state.readiness_report
            readiness_data: Dict[str, Any] = {'report_metadata': {'generated_at': datetime.now().isoformat(), 'session_id': state.session_id, 'version': settings.APP_VERSION}, 'readiness_score': report.readiness_score, 'ready_for_training': report.ready_for_training, 'scoring_formula': {'description': 'Weighted scoring across 6 criteria (total 100 points)', 'weights': {'no_missing_values': 25, 'no_duplicates': 10, 'categorical_encoding': 20, 'numerical_scaling': 15, 'datatype_validity': 15, 'feature_completeness': 15}, 'minimum_passing_score': settings.MIN_READINESS_SCORE}, 'validation_details': {'remaining_missing_values': report.remaining_missing_values, 'remaining_duplicates': report.remaining_duplicates, 'categorical_encoded': report.categorical_encoded, 'numerical_scaled': report.numerical_scaled, 'all_datatypes_valid': report.all_datatypes_valid}, 'data_quality_checks': report.data_quality_checks, 'issues': report.issues, 'recommendations': report.recommendations, 'validation_timestamp': report.validation_timestamp}
            filepath = self.output_dir / 'readiness_report.json'
            filepath.write_text(json.dumps(readiness_data, indent=2, default=str, ensure_ascii=False), encoding='utf-8')
            path = self._save_file(filepath, 'Readiness report')
            return path
        except OutputGenerationException:
            raise
        except Exception as e:
            raise OutputGenerationException(f'Failed to generate readiness report: {str(e)}') from e

    def copy_eda_summary(self, state: PreprocessingGraphState) -> str:
        logger.info('Saving EDA summary...')
        try:
            eda_summary_content = ''
            if state.eda_results and state.eda_results.markdown_summary:
                eda_summary_content = state.eda_results.markdown_summary
            else:
                eda_summary_content = '# EDA Summary\n\nNo EDA results were generated during this processing session.\n'
            filepath = self.output_dir / 'eda_summary.md'
            filepath.write_text(eda_summary_content, encoding='utf-8')
            path = self._save_file(filepath, 'EDA summary')
            return path
        except Exception as e:
            raise OutputGenerationException(f'Failed to save EDA summary: {str(e)}') from e

    def execute(self, state: PreprocessingGraphState) -> List[str]:
        logger.info('=' * 60)
        logger.info('STARTING OUTPUT GENERATION')
        logger.info('=' * 60)
        try:
            cleaned_path = self.export_cleaned_dataset(state)
            state.cleaned_dataset_path = cleaned_path
        except OutputGenerationException as e:
            logger.error(f'  Failed to export cleaned dataset: {str(e)}')
        try:
            processed_path = self.export_processed_dataset(state)
            state.processed_dataset_path = processed_path
        except OutputGenerationException as e:
            logger.error(f'  Failed to export processed dataset: {str(e)}')
        try:
            report_path = self.generate_preprocessing_report(state)
            state.preprocessing_report_path = report_path
        except OutputGenerationException as e:
            logger.error(f'  Failed to generate preprocessing report: {str(e)}')
        try:
            readiness_path = self.generate_readiness_report(state)
            state.readiness_report_path = readiness_path
        except OutputGenerationException as e:
            logger.error(f'  Failed to generate readiness report: {str(e)}')
        try:
            eda_path = self.copy_eda_summary(state)
            state.eda_summary_path = eda_path
        except OutputGenerationException as e:
            logger.error(f'  Failed to save EDA summary: {str(e)}')
        logger.info('=' * 60)
        logger.info('OUTPUT GENERATION COMPLETE')
        logger.info(f'  Total files generated: {len(self.generated_files)}')
        for f in self.generated_files:
            logger.info(f'    - {f}')
        logger.info('=' * 60)
        return self.generated_files

async def output_generator_node(state: PreprocessingGraphState) -> PreprocessingGraphState:
    try:
        logger.info(f'[Node 7] Starting output generation for session {state.session_id}')
        state.add_log('Starting output generation...')
        output_dir = settings.OUTPUTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        generator = OutputGenerator(output_dir)
        logger.info('Generating all output files...')
        state.add_log('Generating CSV exports, JSON reports, and markdown summary...')
        generated_files = generator.execute(state)
        for file_path in generated_files:
            state.add_file(file_path)
        state.execution_status = 'completed'
        state.processing_end_time = datetime.now().isoformat()
        summary = f'Output generation complete: {len(generated_files)} files generated. Pipeline status: {state.execution_status}'
        logger.info(summary)
        state.add_log(summary)
        state.updated_at = datetime.now().isoformat()
        logger.info('[Node 7] Output generation completed successfully')
        return state
    except OutputGenerationException as e:
        logger.error(f'[Node 7] Output generation failed: {str(e)}')
        state.mark_failed(f'Output generation error: {str(e)}')
        state.add_log(f'ERROR: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'[Node 7] Unexpected error in output generation: {str(e)}')
        error_msg = f'Unexpected error during output generation: {str(e)}'
        state.mark_failed(error_msg)
        state.add_log(f'ERROR: {error_msg}')
        raise OutputGenerationException(error_msg) from e