import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle
import tempfile

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

# 全局模型管理器实例字典，按GPU索引存储
model_managers = {}

class MultiGPUModelManager:
    """管理多GPU上的模型实例"""
    
    def __init__(self, devices: List[str]):
        self.devices = devices
        self.model_managers = {}
        # 在初始化时不加载模型，由各进程按需加载
        logger.info(f"初始化了多GPU模型管理器，使用设备: {devices}")
    
    def get_model_manager(self, device: str) -> ModelManager:
        """按需获取特定设备的模型管理器"""
        if device not in self.model_managers:
            self.model_managers[device] = ModelManager(device=device)
            self.model_managers[device].load_all_models()
            logger.info(f"为设备 {device} 加载了模型")
        return self.model_managers[device]

def get_available_gpus() -> List[str]:
    """获取系统中可用的所有GPU"""
    if not torch.cuda.is_available():
        return ["cpu"]
    
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return ["cpu"]
    
    # 检查每个GPU的可用内存
    available_gpus = []
    for i in range(gpu_count):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_mem_gb = free_mem / (1024**3)
            logger.info(f"GPU {i}: 可用内存 {free_mem_gb:.2f} GB")
            
            # 只有当可用内存超过2GB时才添加此GPU
            if free_mem_gb > 2.0:
                available_gpus.append(f"cuda:{i}")
        except Exception as e:
            logger.warning(f"检查GPU {i}内存时出错: {str(e)}")
    
    if not available_gpus:
        return ["cpu"]
    
    return available_gpus

def get_multi_gpu_model_manager(config: Config = Depends(get_config)) -> MultiGPUModelManager:
    global model_managers
    
    if not model_managers:
        devices = get_available_gpus()
        if len(devices) == 1 and devices[0] == "cpu":
            # 如果只有CPU可用，则使用MPS（如果支持）或CPU
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                devices = ["mps"]
            else:
                devices = ["cpu"]
        
        model_managers = MultiGPUModelManager(devices)
        logger.info(f"已初始化多GPU模型管理器，使用设备: {devices}")
    
    return model_managers

def process_frames_worker(args):
    """
    用于进程池的工作函数，处理视频帧的一个子集
    
    Args:
        args: 包含处理所需所有参数的元组
        
    Returns:
        处理后的帧数据列表
    """
    (device, video_path, start_frame, end_frame, batch_size, temp_dir) = args
    
    # 确保在正确的设备上运行
    if device.startswith("cuda:"):
        device_idx = int(device.split(':')[1])
        torch.cuda.set_device(device_idx)
    
    # 创建一个独立的模型管理器和追踪器
    model_manager = ModelManager(device=device)
    model_manager.load_all_models()
    tracker = sv.ByteTrack()
    
    # 获取模型
    pitch_model = model_manager.get_model("pitch")
    player_model = model_manager.get_model("player")
    
    # 创建视频处理器
    video_processor = VideoProcessor(
        device="cpu",  # 视频读取在CPU上进行
        cuda_timeout=10800.0,
        mps_timeout=10800.0,
        cpu_timeout=10800.0
    )
    
    frames_data = []
    current_batch = []
    current_frame_numbers = []
    
    # 处理指定范围的帧
    frame_idx = 0
    frame_stream = video_processor.sync_stream_frames(video_path)
    
    for frame_number, frame in frame_stream:
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        
        if frame_idx >= end_frame:
            break
        
        current_batch.append(frame)
        current_frame_numbers.append(frame_number)
        
        if len(current_batch) >= batch_size or frame_idx == end_frame - 1:
            # 批量处理
            if len(current_batch) > 0:
                try:
                    # 模型推理
                    pitch_results = pitch_model(current_batch, verbose=False)
                    player_results = player_model(current_batch, imgsz=1280, verbose=False)
                    
                    # 处理每一帧的结果
                    for i in range(len(current_batch)):
                        pitch_result = pitch_results[i]
                        player_result = player_results[i]
                        frame_number = current_frame_numbers[i]
                        
                        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
                        
                        detections = sv.Detections.from_ultralytics(player_result)
                        detections = tracker.update_with_detections(detections)
                        
                        # 转换为Python原生类型
                        frame_data = {
                            "frame_number": int(frame_number),
                            "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
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
                                )
                            ] if detections and detections.tracker_id is not None else []
                        }
                        frames_data.append(frame_data)
                except Exception as e:
                    print(f"在设备 {device} 上处理帧批次时出错: {str(e)}")
                
                # 清空批次
                current_batch = []
                current_frame_numbers = []
                
                # 每处理100帧打印一次日志
                if frame_idx % 100 == 0:
                    print(f"设备 {device} 已处理至帧 {frame_idx}/{end_frame-1}")
        
        frame_idx += 1
    
    # 将结果保存到临时文件
    result_file = os.path.join(temp_dir, f"results_{start_frame}_{end_frame}.pkl")
    with open(result_file, 'wb') as f:
        pickle.dump(frames_data, f)
    
    print(f"设备 {device} 完成处理 {start_frame} 到 {end_frame-1}，保存到 {result_file}")
    return result_file

def sync_process_soccer_video(
    video_path: str,
    devices: List[str],
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    使用多进程并行处理足球视频
    
    Args:
        video_path: 视频文件路径
        devices: 可用设备列表
        batch_size: 批处理大小
        
    Returns:
        包含处理结果的字典
    """
    start_time = time.time()
    
    # 创建临时目录存储中间结果
    temp_dir = tempfile.mkdtemp()
    try:
        # 首先获取视频总帧数
        video_processor = VideoProcessor(device="cpu")
        frame_count = 0
        for _ in video_processor.sync_stream_frames(video_path):
            frame_count += 1
        
        print(f"视频总帧数: {frame_count}")
        
        # 根据设备数量划分任务
        frames_per_device = frame_count // len(devices)
        tasks = []
        
        for i, device in enumerate(devices):
            start_frame = i * frames_per_device
            end_frame = start_frame + frames_per_device if i < len(devices) - 1 else frame_count
            print(f"设备 {device} 将处理帧 {start_frame} 到 {end_frame-1}")
            
            # 创建处理任务
            task = (device, video_path, start_frame, end_frame, batch_size, temp_dir)
            tasks.append(task)
        
        # 使用进程池并行处理
        all_result_files = []
        with ProcessPoolExecutor(max_workers=len(devices)) as executor:
            result_futures = executor.map(process_frames_worker, tasks)
            all_result_files = list(result_futures)
        
        # 收集并合并所有结果
        all_frames_data = []
        for result_file in all_result_files:
            with open(result_file, 'rb') as f:
                frames_data = pickle.load(f)
                all_frames_data.extend(frames_data)
        
        # 按帧号排序
        all_frames_data.sort(key=lambda x: x["frame_number"])
        
        # 整合结果
        tracking_data = {
            "frames": all_frames_data,
            "processing_time": time.time() - start_time
        }
        
        total_frames = len(tracking_data["frames"])
        processing_time = tracking_data["processing_time"]
        fps = total_frames / processing_time if processing_time > 0 else 0
        
        print(
            f"完成处理 {total_frames} 帧，耗时 {processing_time:.1f}秒 "
            f"({fps:.2f} fps)，使用设备: {devices}"
        )
        
        return tracking_data
        
    finally:
        # 清理临时文件
        for filename in os.listdir(temp_dir):
            try:
                os.unlink(os.path.join(temp_dir, filename))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

class VideoProcessRequest(BaseModel):
    challenge_id: str
    video_url: str

async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    multi_gpu_manager: MultiGPUModelManager = Depends(get_multi_gpu_model_manager),
):
    logger.info("尝试获取矿工锁...")
    async with miner_lock:
        logger.info("矿工锁已获取，开始处理挑战...")
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"收到挑战数据: {json.dumps(challenge_data, indent=2)}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="未提供视频URL")
            
            logger.info(f"处理挑战 {challenge_id} 的视频 {video_url}")
            
            video_path = await download_video(video_url)
            
            try:
                # 使用多进程同步处理视频（不使用asyncio）
                # 创建一个单独的线程来执行同步处理，避免阻塞事件循环
                loop = asyncio.get_event_loop()
                tracking_data = await loop.run_in_executor(
                    None,
                    sync_process_soccer_video,
                    video_path,
                    multi_gpu_manager.devices,
                    16  # 批处理大小
                )
                
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                
                logger.info(f"完成挑战 {challenge_id}，耗时 {tracking_data['processing_time']:.2f} 秒")
                return response
                
            finally:
                try:
                    os.unlink(video_path)
                except:
                    pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"处理足球挑战时出错: {str(e)}")
            logger.exception("完整错误回溯:")
            raise HTTPException(status_code=500, detail=f"挑战处理错误: {str(e)}")
        finally:
            logger.info("释放矿工锁...")

# 增加同步视频流方法，用于多进程处理
def add_sync_stream_frames_to_video_processor():
    def sync_stream_frames(self, video_path: str):
        """同步版本的视频帧流处理器，用于多进程环境"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            yield frame_number, frame
            frame_number += 1
        
        cap.release()
    
    # 将方法动态添加到VideoProcessor类
    VideoProcessor.sync_stream_frames = sync_stream_frames

# 在导入后立即添加方法
add_sync_stream_frames_to_video_processor()

# 创建带有依赖的路由器
router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)
