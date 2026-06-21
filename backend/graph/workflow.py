import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from backend.graph.state import PreprocessingGraphState
from backend.graph.nodes.dataset_profiling_node import dataset_profiling_node
from backend.graph.nodes.preprocessing_planner_node import preprocessing_planner_node
from backend.graph.nodes.cleaning_executor_node import cleaning_executor_node
from backend.graph.nodes.feature_engineering_node import feature_engineering_node
from backend.graph.nodes.eda_generator_node import eda_generator_node
from backend.graph.nodes.readiness_validator_node import readiness_validator_node
from backend.graph.nodes.output_generator_node import output_generator_node
from backend.utils.exceptions import GraphExecutionException
logger = logging.getLogger(__name__)
PIPELINE_NODES = [('dataset_profiling', dataset_profiling_node), ('preprocessing_planner', preprocessing_planner_node), ('cleaning_executor', cleaning_executor_node), ('feature_engineering', feature_engineering_node), ('eda_generator', eda_generator_node), ('readiness_validator', readiness_validator_node), ('output_generator', output_generator_node)]

class PreprocessingWorkflow:

    def __init__(self):
        self.nodes = PIPELINE_NODES
        self.current_node_index: int = 0
        self.is_running: bool = False

    def _create_initial_state(self, dataset_path: str, session_id: Optional[str]=None) -> PreprocessingGraphState:
        if session_id is None:
            session_id = str(uuid.uuid4())[:12]
        now = datetime.now().isoformat()
        state = PreprocessingGraphState(session_id=session_id, created_at=now, updated_at=now, dataset_path=dataset_path, execution_status='pending')
        logger.info(f'Created initial state for session {session_id}')
        return state

    async def _execute_node(self, node_name: str, node_func: Any, state: PreprocessingGraphState) -> PreprocessingGraphState:
        logger.info(f"{'=' * 60}")
        logger.info(f'EXECUTING NODE: {node_name}')
        logger.info(f"{'=' * 60}")
        start_time = datetime.now()
        state.add_log(f"--- Node '{node_name}' started ---")
        try:
            updated_state = await node_func(state)
            elapsed = (datetime.now() - start_time).total_seconds()
            updated_state.add_log(f"--- Node '{node_name}' completed in {elapsed:.2f}s ---")
            logger.info(f"Node '{node_name}' completed in {elapsed:.2f}s")
            return updated_state
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            error_msg = f"Node '{node_name}' failed after {elapsed:.2f}s: {str(e)}"
            logger.error(error_msg)
            state.add_log(f"--- Node '{node_name}' FAILED: {str(e)} ---")
            raise GraphExecutionException(error_msg) from e

    async def run(self, dataset_path: str, session_id: Optional[str]=None) -> PreprocessingGraphState:
        if self.is_running:
            raise GraphExecutionException('A workflow is already running')
        self.is_running = True
        self.current_node_index = 0
        try:
            state = self._create_initial_state(dataset_path, session_id)
            state.execution_status = 'running'
            state.processing_start_time = datetime.now().isoformat()
            logger.info('=' * 60)
            logger.info('STARTING PREPROCESSING PIPELINE')
            logger.info(f'  Session ID: {state.session_id}')
            logger.info(f'  Dataset: {dataset_path}')
            logger.info(f'  Nodes: {len(self.nodes)}')
            logger.info('=' * 60)
            state.add_log(f'Pipeline started for dataset: {dataset_path}')
            for (index, (node_name, node_func)) in enumerate(self.nodes):
                self.current_node_index = index
                if state.execution_status == 'failed':
                    logger.error(f"Pipeline aborted at node '{node_name}' due to previous failure")
                    state.add_log(f"Pipeline aborted at '{node_name}': {state.error_message}")
                    break
                try:
                    state = await self._execute_node(node_name, node_func, state)
                except GraphExecutionException as e:
                    logger.error(f"Pipeline failed at node '{node_name}': {str(e)}")
                    state.mark_failed(str(e))
                    state.add_log(f"PIPELINE FAILED at '{node_name}': {str(e)}")
                    break
            if state.execution_status != 'failed':
                state.execution_status = 'completed'
                state.processing_end_time = datetime.now().isoformat()
                start = datetime.fromisoformat(state.processing_start_time)
                end = datetime.fromisoformat(state.processing_end_time)
                total_time = (end - start).total_seconds()
                state.add_log(f'Pipeline completed successfully in {total_time:.2f}s')
                logger.info('=' * 60)
                logger.info('PIPELINE COMPLETED SUCCESSFULLY')
                logger.info(f'  Total time: {total_time:.2f}s')
                logger.info(f'  Files generated: {len(state.generated_files)}')
                logger.info(f'  Execution logs: {len(state.execution_logs)}')
                if state.readiness_report:
                    logger.info(f'  ML Readiness Score: {state.readiness_report.readiness_score}/100')
                logger.info('=' * 60)
            else:
                state.processing_end_time = datetime.now().isoformat()
                logger.error('=' * 60)
                logger.error('PIPELINE FAILED')
                logger.error(f'  Error: {state.error_message}')
                logger.error('=' * 60)
            return state
        finally:
            self.is_running = False
            self.current_node_index = 0

    def get_progress(self) -> Dict[str, Any]:
        if not self.is_running:
            return {'is_running': False, 'current_node': None, 'current_index': 0, 'total_nodes': len(self.nodes), 'progress_pct': 0}
        current_name = self.nodes[self.current_node_index][0] if self.current_node_index < len(self.nodes) else 'done'
        progress_pct = int(self.current_node_index / len(self.nodes) * 100)
        return {'is_running': True, 'current_node': current_name, 'current_index': self.current_node_index, 'total_nodes': len(self.nodes), 'progress_pct': progress_pct}
workflow = PreprocessingWorkflow()

async def run_preprocessing_pipeline(dataset_path: str, session_id: Optional[str]=None) -> PreprocessingGraphState:
    return await workflow.run(dataset_path, session_id)