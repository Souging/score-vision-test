import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
import supervision as sv
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

class VideoProcessor:
    """处理视频流，支持CUDA/cuDNN加速的视频处理器"""
    
    def __init__(
        self,
        device: str = "cpu",
        cuda_timeout: float = 900.0,
        mps_timeout: float = 1800.0,
        cpu_timeout: float = 10800.0,
        max_frames_in_memory: int = 100,
        batch_size: int = 10,
        preprocess_options: Optional[Dict[str, Any]] = None,
        use_cuda: bool = False,
        use_cudnn: bool = False
    ):
        self.device = device
        if device == "cuda":
            self.processing_timeout = cuda_timeout
            # 仅当设备是CUDA时才启用CUDA加速
            self.use_cuda = use_cuda
            self.use_cudnn = use_cudnn and use_cuda  # cuDNN依赖于CUDA
        elif device == "mps":
            self.processing_timeout = mps_timeout
            self.use_cuda = False
            self.use_cudnn = False
        else:  # cpu或其他设备
            self.processing_timeout = cpu_timeout
            self.use_cuda = False
            self.use_cudnn = False
        
        # 检查CUDA可用性
        if self.use_cuda and not cv2.cuda.getCudaEnabledDeviceCount():
            logger.warning("CUDA请求但不可用，回退到CPU处理")
            self.use_cuda = False
            self.use_cudnn = False
        
        # 检查cuDNN可用性
        if self.use_cudnn:
            try:
                # 这个检查是不完整的，在实际使用中可能需要更好的cuDNN检测方法
                cv2.dnn.getAvailableBackends()
                has_cudnn = cv2.dnn.DNN_BACKEND_CUDA in cv2.dnn.getAvailableBackends()
                if not has_cudnn:
                    logger.warning("cuDNN请求但不可用，仅使用CUDA")
                    self.use_cudnn = False
            except:
                logger.warning("检查cuDNN可用性时出错，禁用cuDNN")
                self.use_cudnn = False
        
        # 初始化CUDA流（如果使用CUDA）
        self.cuda_stream = None
        if self.use_cuda:
            try:
                self.cuda_stream = cv2.cuda_Stream()
                logger.info("CUDA流初始化成功")
            except Exception as e:
                logger.error(f"初始化CUDA流时出错: {str(e)}")
                self.use_cuda = False
                self.use_cudnn = False
        
        self.max_frames_in_memory = max_frames_in_memory
        self.batch_size = batch_size
        self.preprocess_options = preprocess_options or {}
        self._executor = ThreadPoolExecutor(max_workers=2)
            
        logger.info(
            f"视频处理器初始化: 设备={device}, 超时={self.processing_timeout:.1f}s, "
            f"CUDA={self.use_cuda}, cuDNN={self.use_cudnn}, "
            f"批大小={batch_size}, 最大内存帧数={max_frames_in_memory}"
        )
    
    def __del__(self):
        """确保资源正确释放"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        if hasattr(self, 'cuda_stream') and self.cuda_stream is not None:
            # OpenCV不需要显式释放CUDA流，但记录一下
            logger.debug("CUDA流资源释放")
    
    async def stream_frames(
        self,
        video_path: str,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        异步流式处理视频帧，支持CUDA加速
        
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
                    
                    # 如果使用CUDA，将帧上传到GPU
                    if self.use_cuda:
                        try:
                            gpu_frame = cv2.cuda_GpuMat()
                            gpu_frame.upload(frame)
                            frames.append(gpu_frame)
                        except Exception as e:
                            logger.error(f"上传帧到GPU失败: {str(e)}")
                            # 回退到CPU模式
                            frames.append(frame)
                    else:
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
                batch_limit = min(self.batch_size, self.max_frames_in_memory)
                if max_frames:
                    batch_limit = min(batch_limit, max_frames - frames_processed)
                
                batch = await _read_batch(cap, batch_limit)
                
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
                    
                    # 预处理帧 - 支持CUDA
                    processed_frame = self._preprocess_frame(frame)
                    
                    # 如果是GpuMat，需要下载回CPU
                    if self.use_cuda and isinstance(processed_frame, cv2.cuda_GpuMat):
                        cpu_frame = processed_frame.download()
                        yield current_frame, cpu_frame
                    else:
                        yield current_frame, processed_frame
                        
                    frames_processed += 1
                    
                    # 防止事件循环阻塞
                    if i % 5 == 0:
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
    
    def _preprocess_frame(self, frame):
        """预处理帧，支持CUDA加速"""
        if not self.preprocess_options:
            return frame
            
        if self.use_cuda and isinstance(frame, cv2.cuda_GpuMat):
            # GPU上的处理
            processed = frame  # 在GPU上就不需要复制了
            
            # 应用预处理选项
            if 'resize' in self.preprocess_options:
                width, height = self.preprocess_options['resize']
                processed = cv2.cuda.resize(processed, (width, height), stream=self.cuda_stream)
                
            if 'crop' in self.preprocess_options:
                x, y, w, h = self.preprocess_options['crop']
                processed = processed.rowRange(y, y+h).colRange(x, x+w)
                
            if 'normalize' in self.preprocess_options and self.preprocess_options['normalize']:
                # CUDA上的归一化
                processed = cv2.cuda.divide(processed, 255.0, stream=self.cuda_stream)
                
            return processed
        else:
            # CPU上的处理
            processed = frame.copy() if not isinstance(frame, cv2.cuda_GpuMat) else frame.download().copy()
            
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

    # 支持CUDA加速的其他方法
    def apply_cuda_filter(self, gpu_frame, filter_type):
        """应用CUDA加速的滤镜"""
        if not self.use_cuda or not isinstance(gpu_frame, cv2.cuda_GpuMat):
            return gpu_frame
            
        try:
            if filter_type == 'blur':
                return cv2.cuda.blur(gpu_frame, (5, 5), stream=self.cuda_stream)
            elif filter_type == 'gaussian':
                return cv2.cuda.GaussianBlur(gpu_frame, (5, 5), 0, stream=self.cuda_stream)
            elif filter_type == 'median':
                return cv2.cuda.medianFilter(gpu_frame, 5, stream=self.cuda_stream)
            else:
                return gpu_frame
        except Exception as e:
            logger.error(f"应用CUDA滤镜失败: {str(e)}")
            return gpu_frame

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
