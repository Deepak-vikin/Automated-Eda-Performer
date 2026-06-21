import logging
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.config import settings
from backend.api.routes import router as api_router
from backend.utils.logger import setup_logging
logger = setup_logging()

def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, description='AI-Powered Data Preprocessing and AutoEDA System. Upload a dataset and get it automatically profiled, cleaned, feature-engineered, and validated for ML readiness.', docs_url='/docs', redoc_url='/redoc')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    app.include_router(api_router, prefix='/api', tags=['AutoEDA API'])
    frontend_dir = settings.BASE_DIR / 'frontend'
    if frontend_dir.exists():
        app.mount('/css', StaticFiles(directory=str(frontend_dir / 'css')), name='css')
        app.mount('/js', StaticFiles(directory=str(frontend_dir / 'js')), name='js')
    outputs_dir = settings.OUTPUTS_DIR
    if outputs_dir.exists():
        app.mount('/outputs', StaticFiles(directory=str(outputs_dir)), name='outputs')

    @app.get('/', include_in_schema=False)
    async def serve_index():
        index_path = frontend_dir / 'index.html'
        if index_path.exists():
            return FileResponse(str(index_path))
        return {'message': 'AutoEDA API is running. Visit /docs for API documentation.'}

    @app.get('/results.html', include_in_schema=False)
    async def serve_results():
        results_path = frontend_dir / 'results.html'
        if results_path.exists():
            return FileResponse(str(results_path))
        return {'message': 'Results page not found'}

    @app.on_event('startup')
    async def startup_event():
        logger.info('=' * 60)
        logger.info(f'Starting {settings.APP_NAME} v{settings.APP_VERSION}')
        logger.info('=' * 60)
        settings.validate_directories()
        logger.info(f'  Base directory:    {settings.BASE_DIR}')
        logger.info(f'  Uploads directory: {settings.UPLOADS_DIR}')
        logger.info(f'  Outputs directory: {settings.OUTPUTS_DIR}')
        logger.info(f'  EDA output:        {settings.EDA_OUTPUT_DIR}')
        logger.info(f'  Ollama URL:        {settings.OLLAMA_BASE_URL}')
        logger.info(f'  Ollama model:      {settings.OLLAMA_MODEL}')
        logger.info(f'  Debug mode:        {settings.DEBUG}')
        logger.info('=' * 60)

    @app.on_event('shutdown')
    async def shutdown_event():
        logger.info('AutoEDA application shutting down...')
    return app
app = create_app()
if __name__ == '__main__':
    uvicorn.run('backend.main:app', host=settings.HOST, port=settings.PORT, reload=settings.DEBUG, workers=settings.WORKERS, log_level=settings.LOG_LEVEL.lower())