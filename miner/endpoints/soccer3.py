import os
import json
import time
from typing import Optional, Dict, Any, List  # 添加List导入
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Global model manager instance
model_manager = None

def get_model_manager() -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device="cuda")
        model_manager.load_all_models()
    return model_manager
model_manager = get_model_manager()
async def process_soccer_video(
    video_path: str,
    model_manager: ModelManager,
    batch_size: int = 32,  # 可调整的批处理大小（根据GPU显存调整）
) -> Dict[str, Any]:
    """优化版：使用批处理加速足球视频分析"""
    start_time = time.time()
    tracking_data = {"frames": []}
    
    try:
        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0
        )
        
        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(status_code=400, detail="视频文件不可读或已损坏")
        # 预加载模型
        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")
        tracker = sv.ByteTrack()
        # 异步批处理帧
        batch_frames = []
        batch_indices = []
        
        async for frame_number, frame in video_processor.stream_frames(video_path):
            batch_frames.append(frame)
            batch_indices.append(frame_number)
            
            # 当累积到batch_size时处理一批次
            if len(batch_frames) == batch_size:
                await _process_batch(
                    batch_frames, batch_indices,
                    player_model, pitch_model, tracker,
                    tracking_data
                )
                batch_frames.clear()
                batch_indices.clear()
        # 处理剩余不足batch_size的帧
        if batch_frames:
            await _process_batch(
                batch_frames, batch_indices,
                player_model, pitch_model, tracker,
                tracking_data
            )
        # 性能统计
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        total_frames = len(tracking_data["frames"])
        
        logger.info(
            f"处理完成 | 总帧数: {total_frames} | "
            f"耗时: {processing_time:.1f}s | "
            f"平均FPS: {total_frames/max(0.1, processing_time):.1f} | "
            f"设备: {model_manager.device}"
        )
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"视频处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"视频处理错误: {str(e)}")
async def _process_batch(
    frames: List[np.ndarray],
    frame_numbers: List[int],
    player_model,
    pitch_model,
    tracker,
    tracking_data: Dict[str, Any]
):
    """处理一个批次的帧（核心优化逻辑）"""
    try:
        # 并行执行模型推理
        player_results = player_model(frames, imgsz=1280, verbose=False)
        pitch_results = pitch_model(frames, verbose=False)
        
        # 逐帧处理追踪结果
        for i, (frame_num, player_res, pitch_res) in enumerate(zip(
            frame_numbers, player_results, pitch_results
        )):
            # 球场关键点检测
            keypoints = sv.KeyPoints.from_ultralytics(pitch_res)
            
            # 球员检测与追踪
            detections = sv.Detections.from_ultralytics(player_res)
            detections = tracker.update_with_detections(detections)
            
            tracking_data["frames"].append({
                "frame_number": int(frame_num),
                "keypoints": keypoints.xy[0].tolist() if keypoints.xy.any() else [],
                "objects": [
                    {
                        "id": int(tracker_id),
                        "bbox": [float(x) for x in bbox],
                        "class_id": int(class_id)
                    }
                    for tracker_id, bbox, class_id in zip(
                        detections.tracker_id,
                        detections.xyxy,
                        detections.class_id
                    ) if tracker_id is not None
                ]
            })
        # 每处理完100帧打印进度
        if frame_numbers[-1] % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"进度: {frame_numbers[-1]}帧 | "
                f"批处理速度: {len(frames)/max(0.1, elapsed):.1f} fps"
            )
            
    except Exception as e:
        logger.error(f"批处理失败: {str(e)}")
        raise


async def process_challenge(
    request: Request,
    #config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager),
):
    logger.info("Attempting to acquire miner lock...")
    async with miner_lock:
        logger.info("Miner lock acquired, processing challenge...")
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            
            logger.info(f"Processing challenge {challenge_id} with video {video_url}")
            
            video_path = await download_video(video_url)
            
            try:
                tracking_data = await process_soccer_video(
                    video_path,
                    model_manager
                )
                
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                
                logger.info(f"Completed challenge {challenge_id} in {tracking_data['processing_time']:.2f} seconds")
                return response
                
            finally:
                try:
                    model_manager.clear_cache()
                    os.unlink(video_path)
                except:
                    pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
        finally:
            logger.info("Releasing miner lock...")

# Create router with dependencies
router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    #dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)
