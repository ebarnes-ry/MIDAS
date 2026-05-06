"""
Trajectory inspection endpoints.
Provides read access to logged trajectories for analysis and portfolio demos.
"""
from fastapi import APIRouter, Depends, Query
from src.models.manager import ModelManager
from ..dependencies.session import get_model_manager
from src.pipeline.trajectory import TrajectoryLogger

router = APIRouter()


def get_logger(model_manager: ModelManager = Depends(get_model_manager)) -> TrajectoryLogger:
    log_path = model_manager.config.get("trajectory_log_path", "trajectories/midas_trajectories.jsonl")
    return TrajectoryLogger(log_path=log_path)


@router.get("/")
async def list_trajectories(
    n: int = Query(default=20, le=100),
    logger: TrajectoryLogger = Depends(get_logger),
):
    """Return the N most recent trajectories."""
    return {"trajectories": logger.read_trajectories(n=n)}


@router.get("/stats")
async def trajectory_stats(logger: TrajectoryLogger = Depends(get_logger)):
    """Summary statistics over all logged trajectories."""
    return logger.get_stats()
