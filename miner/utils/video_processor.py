import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
import supervision as sv
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

class VideoProcessor:
    """处理视频流，支持批处理和内存管理的视频处理器"""
    
    def __init__(
        self,
        device: str = "cpu",
        cuda_timeout: float = 900.0,
        mps_timeout: float = 1800.0,
        cpu_timeout: float = 10800.0,
        max_frames_in_memory: int = 750,  # 内存中保存的最大帧数
        batch_size: int = 15,  # 批量读取帧数
        preprocess_options: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        # 基于设备设置超时
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:  # cpu或其他设备
            self.processing_timeout = cpu_timeout
        
        self.max_frames_in_memory = max_frames_in_memory
        self.batch_size = batch_size
        self.preprocess_options = preprocess_options or {}
        self._executor = ThreadPoolExecutor(max_workers=2)  # 限制线程数
            
        logger.info(
            f"视频处理器初始化: 设备={device}, 超时={self.processing_timeout:.1f}s, "
            f"批大小={batch_size}, 最大内存帧数={max_frames_in_memory}"
        )
    
    def __del__(self):
        """确保线程池正确关闭"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    async def stream_frames(
        self,
        video_path: str,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        异步流式处理视频帧，支持批处理和内存管理
        
        Args:
            video_path: 视频文件路径
            start_frame: 开始处理的帧索引
            max_frames: 最大处理帧数，None表示处理所有帧
            target_fps: 目标帧率，None表示处理所有帧
            
        Yields:
            Tuple[int, np.ndarray]: 帧号和帧数据
        """
        start_time = time.time()
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = 0
        if target_fps and target_fps < original_fps:
            frame_skip = int(original_fps / target_fps) - 1
            logger.info(f"原始FPS: {original_fps}, 目标FPS: {target_fps}, 将跳过每{frame_skip+1}帧中的{frame_skip}帧")
        
        # 跳到起始帧
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info(f"从第{start_frame}帧开始处理")
        
        try:
            frame_count = start_frame
            frames_processed = 0
            skip_counter = 0
            
            async def _read_batch(cap, batch_size):
                """批量读取视频帧"""
                frames = []
                for _ in range(batch_size):
                    ret, frame = await asyncio.get_event_loop().run_in_executor(
                        self._executor, cap.read
                    )
                    if not ret:
                        break
                    frames.append(frame)
                return frames
            
            while True:
                # 检查超时
                elapsed_time = time.time() - start_time
                if elapsed_time > self.processing_timeout:
                    logger.warning(
                        f"视频处理超时，已用时{elapsed_time:.1f}秒，"
                        f"设备: {self.device}，已处理{frames_processed}帧"
                    )
                    break
                
                # 检查最大帧数限制
                if max_frames and frames_processed >= max_frames:
                    logger.info(f"已达到最大帧数限制({max_frames})，停止处理")
                    break
                
                # 批量读取帧
                batch = await _read_batch(cap, min(self.batch_size, 
                                                  self.max_frames_in_memory,
                                                  max_frames - frames_processed if max_frames else self.batch_size))
                
                if not batch:
                    logger.info(f"处理完成: {frames_processed}帧，用时{elapsed_time:.1f}秒，设备: {self.device}")
                    break
                
                # 处理每一帧
                for i, frame in enumerate(batch):
                    current_frame = frame_count + i
                    
                    # 帧跳过逻辑
                    if frame_skip > 0:
                        skip_counter = (skip_counter + 1) % (frame_skip + 1)
                        if skip_counter != 0:
                            continue
                    
                    # 预处理帧
                    processed_frame = self._preprocess_frame(frame)
                    
                    yield current_frame, processed_frame
                    frames_processed += 1
                    
                    # 防止事件循环阻塞
                    if i % 5 == 0:  # 每5帧让出一次控制权
                        await asyncio.sleep(0)
                
                frame_count += len(batch)
                
                # 定期进行垃圾回收
                if frames_processed % 100 == 0:
                    import gc
                    gc.collect()
        
        except Exception as e:
            logger.error(f"视频处理错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        finally:
            cap.release()
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧，应用配置的预处理选项"""
        if not self.preprocess_options:
            return frame
            
        processed = frame.copy()  # 避免修改原始帧
        
        # 应用预处理选项
        if 'resize' in self.preprocess_options:
            width, height = self.preprocess_options['resize']
            processed = cv2.resize(processed, (width, height))
            
        if 'crop' in self.preprocess_options:
            x, y, w, h = self.preprocess_options['crop']
            processed = processed[y:y+h, x:x+w]
            
        if 'normalize' in self.preprocess_options and self.preprocess_options['normalize']:
            processed = processed / 255.0
            
        return processed
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """获取视频详细信息"""
        try:
            sv_info = sv.VideoInfo.from_video_path(video_path)
            
            # 获取更多信息
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
                
            # 获取更多视频属性
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            return {
                "width": sv_info.width,
                "height": sv_info.height,
                "fps": sv_info.fps,
                "total_frames": sv_info.total_frames,
                "codec": codec_str,
                "duration_seconds": sv_info.total_frames / sv_info.fps if sv_info.fps > 0 else 0
            }
        except Exception as e:
            logger.error(f"获取视频信息错误: {str(e)}")
            return {}
    
    @staticmethod
    async def ensure_video_readable(video_path: str, timeout: float = 5.0) -> bool:
        """
        检查视频是否可读，带超时保护
        
        Args:
            video_path: 视频文件路径
            timeout: 检查视频的最大等待时间
            
        Returns:
            bool: 如果视频可读则返回True
        """
        try:
            async def _check_video():
                # 更全面的视频检查
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return False
                    
                # 检查能否读取第一帧
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    return False
                    
                # 检查能否获取基本属性
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                cap.release()
                
                # 确保基本属性有效
                return fps > 0 and width > 0 and height > 0
            
            return await asyncio.wait_for(_check_video(), timeout)
        
        except asyncio.TimeoutError:
            logger.error(f"检查视频可读性超时: {video_path}")
            return False
        except Exception as e:
            logger.error(f"检查视频可读性错误: {str(e)}")
            return False
            
    async def extract_keyframes(self, video_path: str, max_keyframes: int = 10) -> List[Tuple[int, np.ndarray]]:
        """提取视频关键帧"""
        keyframes = []
        try:
            # 获取视频总帧数
            info = self.get_video_info(video_path)
            if not info:
                return []
                
            total_frames = info.get("total_frames", 0)
            if total_frames <= 0:
                return []
                
            # 均匀提取关键帧
            step = max(1, total_frames // max_keyframes)
            frame_indices = [i * step for i in range(max_keyframes)]
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []
                
            try:
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        keyframes.append((idx, frame))
            finally:
                cap.release()
                
            return keyframes
            
        except Exception as e:
            logger.error(f"提取关键帧错误: {str(e)}")
            return []
