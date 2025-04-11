#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from pathlib import Path
import time
from loguru import logger
from typing import List, Dict, Union
import concurrent.futures
import multiprocessing as mp  # Import multiprocessing

miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)

from utils.model_manager import ModelManager
from utils.video_downloader import download_video
from endpoints.soccer1 import MultiGPUModelManager, get_available_gpus, sync_process_soccer_video
from utils.device import get_optimal_device
from scripts.download_models import download_models

TEST_VIDEO_URL = "https://pub-a55bd0dbae3c4afd86bd066961ab7d1e.r2.dev/2025_03_23/f2ef17/h1_13e1e0.mp4"
#TEST_VIDEO_URL = "https://pub-a55bd0dbae3c4afd86bd066961ab7d1e.r2.dev/test_10secs.mov"

def optimize_coordinates(coords: List[float]) -> List[float]:
    return [round(float(x), 2) for x in coords]

def filter_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    return [optimize_coordinates(kp) for kp in keypoints if not (kp[0] == 0 and kp[1] == 0)]

def optimize_frame_data(frame_data: Dict) -> Dict:
    optimized_data = {}
    
    if "objects" in frame_data:
        optimized_data["objects"] = []
        for obj in frame_data["objects"]:
            optimized_obj = obj.copy()
            if "bbox" in obj:
                optimized_obj["bbox"] = optimize_coordinates(obj["bbox"])
            optimized_data["objects"].append(optimized_obj)
    
    if "keypoints" in frame_data:
        optimized_data["keypoints"] = filter_keypoints(frame_data["keypoints"])
    
    # 确保帧号被保留
    if "frame_number" in frame_data:
        optimized_data["frame_number"] = frame_data["frame_number"]
    
    return optimized_data

def optimize_result_data(result: Dict[str, Union[Dict, List, float, str]]) -> Dict[str, Union[Dict, List, float, str]]:
    optimized_result = result.copy()
    
    if "frames" in result:
        frames = result["frames"]
        
        if isinstance(frames, list):
            optimized_frames = {}
            for frame_data in frames:
                if frame_data:
                    frame_num = str(frame_data.get("frame_number", 0))
                    optimized_frames[frame_num] = optimize_frame_data(frame_data)
        elif isinstance(frames, dict):
            optimized_frames = {}
            for frame_num, frame_data in frames.items():
                if frame_data:
                    optimized_frames[str(frame_num)] = optimize_frame_data(frame_data)
        else:
            logger.warning(f"Unexpected frames data type: {type(frames)}")
            optimized_frames = frames
            
        optimized_result["frames"] = optimized_frames
    
    if "processing_time" in result:
        optimized_result["processing_time"] = round(float(result["processing_time"]), 2)
    
    return optimized_result

async def main():
    mp.set_start_method('spawn') # or 'forkserver' # 添加这一行

    try:
        logger.info("Starting video processing test")
        start_time = time.time()
        
        logger.info("Checking for required models...")
        download_models()
        
        logger.info(f"Downloading test video from {TEST_VIDEO_URL}")
        video_path = await download_video(TEST_VIDEO_URL)
        logger.info(f"Video downloaded to {video_path}")
        
        available_devices = []
        
        try:
            # 获取可用的所有GPU
            available_devices = get_available_gpus()
            logger.info(f"Available devices: {available_devices}")
            
            # 如果没有可用的GPU，则使用最优设备
            if not available_devices or (len(available_devices) == 1 and available_devices[0] == "cpu"):
                device = get_optimal_device()
                available_devices = [device]
                logger.info(f"No GPUs found, using optimal device: {device}")
            
            # 初始化多GPU模型管理器
            logger.info("Initializing multi-GPU model manager...")
            multi_gpu_manager = MultiGPUModelManager(available_devices)
            
            logger.info("Models loaded successfully on all devices")
            
            # 使用线程池执行器运行同步处理函数
            logger.info("Starting video processing with multiple GPUs...")
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(
                    pool,
                    sync_process_soccer_video,
                    str(video_path),
                    available_devices,
                    48  # 批处理大小
                )
            
            logger.info("Optimizing frame data...")
            optimized_result = optimize_result_data(result)
            
            output_dir = Path(__file__).parent.parent / "test_outputs"
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"pipeline_test_results_{int(time.time())}.json"
            
            result_json = json.dumps(optimized_result)
            data_size = len(result_json) / 1024
            logger.info(f"Result data size: {data_size:.2f} KB")
            
            with open(output_file, "w") as f:
                f.write(result_json)
            
            total_time = time.time() - start_time
            frames = len(optimized_result["frames"])
            fps = frames / optimized_result["processing_time"]
            
            logger.info("Processing completed successfully!")
            logger.info(f"Total frames processed: {frames}")
            logger.info(f"Processing time: {optimized_result['processing_time']:.2f} seconds")
            logger.info(f"Average FPS: {fps:.2f}")
            logger.info(f"Total time (including download): {total_time:.2f} seconds")
            logger.info(f"Results saved to: {output_file}")
            
        finally:
            # 清理所有设备上的模型缓存
            if 'multi_gpu_manager' in locals():
                for device in available_devices:
                    try:
                        model_manager = multi_gpu_manager.get_model_manager(device)
                        model_manager.clear_cache()
                        logger.info(f"Cleared model cache on device {device}")
                    except Exception as e:
                        logger.error(f"Error clearing model cache on device {device}: {e}")
            
    finally:
        try:
            os.unlink(str(video_path))
            logger.info("Cleaned up temporary video file")
        except Exception as e:
            logger.error(f"Error cleaning up video file: {e}")

if __name__ == "__main__":
    asyncio.run(main())
