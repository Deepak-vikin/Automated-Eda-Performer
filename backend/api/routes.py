import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict
from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from backend.config import settings
from backend.graph.state import PreprocessingGraphState
from backend.graph.workflow import PreprocessingWorkflow, workflow
from backend.models.schemas import HealthResponse, ProcessRequest, ProcessResponse, ResultsResponse, UploadResponse
logger = logging.getLogger(__name__)
router = APIRouter()
_session_results: Dict[str, PreprocessingGraphState] = {}
_session_status: Dict[str, str] = {}
_session_files: Dict[str, str] = {}
#Async func -No blockage of other functions while this is running.
    #User uploads a file.
    #Data preprocessing takes 2 minutes.
    #Without async, server waits 2 minutes.
    #With async, server can handle other requests meanwhile.

async def _run_pipeline_task(session_id: str, dataset_path: str) -> None:
    try:
        logger.info(f'Background pipeline started for session {session_id}')
        _session_status[session_id] = 'running'
        pipeline = PreprocessingWorkflow()
        final_state = await pipeline.run(dataset_path=dataset_path, session_id=session_id)
        _session_results[session_id] = final_state
        _session_status[session_id] = final_state.execution_status
        logger.info(f'Background pipeline completed for session {session_id}: {final_state.execution_status}')
    except Exception as e:
        logger.error(f'Background pipeline failed for session {session_id}: {str(e)}')
        _session_status[session_id] = 'failed'

@router.post('/upload', response_model=UploadResponse)
async def upload_dataset(file: UploadFile=File(...)) -> UploadResponse:
    logger.info(f'Upload request received: {file.filename}')
    if not file.filename:
        raise HTTPException(status_code=400, detail='No filename provided')
    file_ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if file_ext not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{file_ext}. Allowed types: {', '.join(settings.ALLOWED_FILE_TYPES)}")
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to read file: {str(e)}')
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=413, detail=f'File too large: {file_size_mb:.1f} MB. Maximum: {settings.MAX_UPLOAD_SIZE_MB} MB')
    session_id = str(uuid.uuid4())[:12]
    upload_dir = settings.UPLOADS_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    session_dir = upload_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / file.filename
    try:
        file_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save file: {str(e)}')
    _session_files[session_id] = str(file_path.resolve())
    _session_status[session_id] = 'uploaded'
    logger.info(f'File uploaded: {file.filename} ({file_size_mb:.2f} MB) -> session {session_id}')
    return UploadResponse(success=True, message='File uploaded successfully', session_id=session_id, filename=file.filename, file_size_mb=round(file_size_mb, 2))

@router.post('/process', response_model=ProcessResponse)
async def start_processing(request: ProcessRequest, background_tasks: BackgroundTasks) -> ProcessResponse:
    session_id = request.session_id
    logger.info(f'Process request received for session {session_id}')
    if session_id not in _session_files:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Upload a file first.")
    current_status = _session_status.get(session_id, 'unknown')
    if current_status == 'running':
        raise HTTPException(status_code=409, detail='Processing is already running for this session')
    dataset_path = _session_files[session_id]
    if not Path(dataset_path).exists():
        raise HTTPException(status_code=404, detail='Uploaded file no longer found on disk')
    _session_status[session_id] = 'queued'
    background_tasks.add_task(_run_pipeline_task, session_id, dataset_path)
    logger.info(f'Processing queued for session {session_id}')
    return ProcessResponse(success=True, message='Processing started. Use GET /results to check progress.', session_id=session_id)

@router.get('/results', response_model=ResultsResponse)
async def get_results(session_id: str) -> ResultsResponse:
    logger.info(f'Results request for session {session_id}')
    if session_id not in _session_status:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    status = _session_status[session_id]
    if status in ('uploaded', 'queued', 'running'):
        return ResultsResponse(success=True, session_id=session_id, execution_logs=[f'Status: {status}'])
    if session_id not in _session_results:
        return ResultsResponse(success=False, session_id=session_id, execution_logs=[f'Status: {status}', 'No results available'])
    state = _session_results[session_id]
    return ResultsResponse(success=state.execution_status == 'completed', session_id=session_id, profile=state.dataset_profile, plan=state.preprocessing_plan, cleaning_actions=state.cleaning_actions, eda_results=state.eda_results, readiness_report=state.readiness_report, generated_files=state.generated_files, execution_logs=state.execution_logs)

@router.get('/download/cleaned')
async def download_cleaned(session_id: str):
    logger.info(f'Download cleaned request for session {session_id}')
    state = _session_results.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail='Session not found or not completed')
    file_path = state.cleaned_dataset_path
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail='Cleaned dataset not available')
    return FileResponse(path=file_path, filename='cleaned_dataset.csv', media_type='text/csv')

@router.get('/download/processed')
async def download_processed(session_id: str):
    logger.info(f'Download processed request for session {session_id}')
    state = _session_results.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail='Session not found or not completed')
    file_path = state.processed_dataset_path
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail='Processed dataset not available')
    return FileResponse(path=file_path, filename='processed_dataset.csv', media_type='text/csv')

@router.get('/download/report')
async def download_report(session_id: str):
    logger.info(f'Download report request for session {session_id}')
    state = _session_results.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail='Session not found or not completed')
    file_path = state.preprocessing_report_path
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail='Preprocessing report not available')
    return FileResponse(path=file_path, filename='preprocessing_report.json', media_type='application/json')

@router.get('/health', response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status='healthy', timestamp=datetime.now().isoformat(), version=settings.APP_VERSION)

@router.get('/status')
async def get_status(session_id: str) -> Dict:
    if session_id not in _session_status:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    status = _session_status[session_id]
    response = {'session_id': session_id, 'status': status, 'progress': workflow.get_progress() if status == 'running' else None}
    if session_id in _session_results:
        state = _session_results[session_id]
        if state.readiness_report:
            response['readiness_score'] = state.readiness_report.readiness_score
            response['ready_for_training'] = state.readiness_report.ready_for_training
    return response