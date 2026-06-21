import logging
from datetime import datetime
from backend.graph.state import PreprocessingGraphState
from backend.services.dataset_profiler import DatasetProfiler
from backend.utils.exceptions import DatasetProfilingException
logger = logging.getLogger(__name__)

async def dataset_profiling_node(state: PreprocessingGraphState) -> PreprocessingGraphState:
    try:
        logger.info(f'[Node 1] Starting dataset profiling for session {state.session_id}')
        state.add_log('Starting dataset profiling...')
        if not state.dataset_path:
            raise DatasetProfilingException('No dataset path provided in state')
        profiler = DatasetProfiler()
        logger.info(f'Loading dataset from {state.dataset_path}')
        df = profiler.load_dataset(state.dataset_path)
        state.add_log(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')
        logger.info('Generating dataset profile...')
        profile = profiler.profile_dataset()
        state.add_log('Dataset profiling completed')
        state.dataset_profile = profile
        state.original_dataframe = df.to_dict(orient='split')
        state.file_type = profile.file_type
        logger.info(f'Profile Summary:')
        logger.info(f'  - Rows: {profile.row_count}')
        logger.info(f'  - Columns: {profile.column_count}')
        logger.info(f'  - Numerical columns: {len(profile.numerical_columns)}')
        logger.info(f'  - Categorical columns: {len(profile.categorical_columns)}')
        logger.info(f'  - DateTime columns: {len(profile.datetime_columns)}')
        logger.info(f'  - Missing values: {profile.total_missing_values}')
        logger.info(f'  - Duplicate rows: {profile.total_duplicate_rows}')
        logger.info(f'  - Memory usage: {profile.memory_usage_mb} MB')
        state.add_log(f'Profile generated: {profile.column_count} columns, {profile.total_missing_values} missing values, {profile.total_duplicate_rows} duplicates')
        state.updated_at = datetime.now().isoformat()
        logger.info('[Node 1] Dataset profiling completed successfully')
        return state
    except DatasetProfilingException as e:
        logger.error(f'[Node 1] Dataset profiling failed: {str(e)}')
        state.mark_failed(f'Dataset profiling error: {str(e)}')
        state.add_log(f'ERROR: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'[Node 1] Unexpected error in dataset profiling: {str(e)}')
        error_msg = f'Unexpected error during profiling: {str(e)}'
        state.mark_failed(error_msg)
        state.add_log(f'ERROR: {error_msg}')
        raise DatasetProfilingException(error_msg) from e