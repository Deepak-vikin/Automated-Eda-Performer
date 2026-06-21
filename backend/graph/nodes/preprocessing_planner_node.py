import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from backend.graph.state import PreprocessingGraphState
from backend.models.schemas import PreprocessingPlan, PreprocessingAction
from backend.utils.exceptions import AIPreprocessingPlanException
logger = logging.getLogger(__name__)

def _generate_rule_based_strategy(state: PreprocessingGraphState) -> Dict[str, Any]:
    strategy = {}
    features_to_remove = []
    target_column = None
    if not state.dataset_profile:
        return {'preprocessing_strategy': {}, 'features_to_remove': []}
    profile = state.dataset_profile
    for col in profile.columns:
        col_strategy = {'missing': None, 'outliers': None, 'encoding': None, 'scaling': None, 'datatype': None, 'remove': False, 'date_decompose': None, 'normalization': False, 'reasoning': 'Rule-based selection.'}
        if col.missing_percentage > 50 or col.unique_count <= 1:
            col_strategy['remove'] = True
            col_strategy['reasoning'] = 'Removed due to >50% missing values or zero variance.'
            features_to_remove.append(col.name)
            strategy[col.name] = col_strategy
            continue
        is_numeric = col.name in profile.numerical_columns
        is_categorical = col.name in profile.categorical_columns or col.data_type == 'object'
        is_datetime = col.name in profile.datetime_columns
        if col.missing_count > 0:
            if is_numeric:
                if col.missing_percentage < 20:
                    col_strategy['missing'] = 'median'
                else:
                    col_strategy['missing'] = 'mean'
            else:
                col_strategy['missing'] = 'mode'
        if is_numeric:
            col_strategy['outliers'] = 'iqr'
            col_strategy['scaling'] = 'standard'
        if is_categorical:
            if col.unique_count <= 10:
                col_strategy['encoding'] = 'one_hot'
            else:
                col_strategy['encoding'] = 'label'
        if is_datetime:
            col_strategy['date_decompose'] = ['year', 'month', 'day', 'weekday']
        strategy[col.name] = col_strategy
    return {'preprocessing_strategy': strategy, 'features_to_remove': features_to_remove, 'target_column': target_column}

def _convert_strategy_to_actions(strategy: Dict[str, Any], state: PreprocessingGraphState) -> List[PreprocessingAction]:
    actions = []
    if not state.dataset_profile:
        return actions
    preprocessing_strategy = strategy.get('preprocessing_strategy', {})
    for (col_name, col_strategy) in preprocessing_strategy.items():
        action = PreprocessingAction(column_name=col_name, missing_strategy=col_strategy.get('missing'), outlier_strategy=col_strategy.get('outliers'), encoding_strategy=col_strategy.get('encoding'), scaling_strategy=col_strategy.get('scaling'), datatype_conversion=col_strategy.get('datatype'), feature_removal=col_strategy.get('remove', False), date_decomposition=col_strategy.get('date_decompose'), normalization=col_strategy.get('normalization', False), reasoning=col_strategy.get('reasoning', ''))
        actions.append(action)
    return actions

async def preprocessing_planner_node(state: PreprocessingGraphState) -> PreprocessingGraphState:
    try:
        logger.info(f'[Node 2] Starting AI preprocessing planning for session {state.session_id}')
        state.add_log('Starting preprocessing planning...')
        if not state.dataset_profile:
            raise AIPreprocessingPlanException('Dataset profile not found in state')
        logger.info('Generating rule-based strategy...')
        strategy = _generate_rule_based_strategy(state)
        actions = _convert_strategy_to_actions(strategy, state)
        features_to_remove = strategy.get('features_to_remove', [])
        summary = 'Applied rule-based preprocessing: imputation, encoding, and scaling based on data types and missing percentages.'
        plan = PreprocessingPlan(plan_timestamp=datetime.now().isoformat(), total_columns=state.dataset_profile.column_count, actions=actions, features_to_remove=features_to_remove, target_column=strategy.get('target_column'), summary=summary)
        state.preprocessing_plan = plan
        logger.info(f'Preprocessing plan created: {len(actions)} actions')
        state.add_log(f'Preprocessing plan created: {len(actions)} actions, {len(features_to_remove)} features to remove')
        state.add_log(f'Plan summary: {summary}')
        state.updated_at = datetime.now().isoformat()
        logger.info('[Node 2] Preprocessing planning completed successfully')
        return state
    except Exception as e:
        logger.error(f'[Node 2] Unexpected error in preprocessing planning: {str(e)}')
        error_msg = f'Unexpected error during preprocessing planning: {str(e)}'
        state.mark_failed(error_msg)
        state.add_log(f'ERROR: {error_msg}')
        raise AIPreprocessingPlanException(error_msg) from e