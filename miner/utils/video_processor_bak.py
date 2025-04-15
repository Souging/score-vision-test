import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple
import cv2
import numpy as np
import supervision as sv
from loguru import logger

class VideoProcessor:
    """Handles video processing with frame streaming, resolution reduction, and frame skipping."""
    
    def __init__(
        self,
        device: str = "cpu",
        cuda_timeout: float = 900.0,  # 15 minutes for CUDA
        mps_timeout: float = 1800.0,  # 30 minutes for MPS
        cpu_timeout: float = 10800.0,  # 3 hours for CPU
        frame_skip: int = 0,  # Process every frame by default
        resolution_factor: float = 1,  # Full resolution by default
        use_keyframes: bool = True,  # Use keyframe detection
        scene_threshold: float = 0.05,  # Threshold for scene change detection
    ):
        self.device = device
        # Set timeout based on device
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:  # cpu or any other device
            self.processing_timeout = cpu_timeout
        
        # Frame processing parameters
        self.frame_skip = max(0, frame_skip)  # Ensure non-negative
        self.resolution_factor = max(0.1, min(1.0, resolution_factor))  # Between 0.1 and 1.0
        self.use_keyframes = use_keyframes
        self.scene_threshold = scene_threshold
        self.prev_frame = None  # Store previous frame for scene change detection
            
        logger.info(f"Video processor initialized with {device} device, timeout: {self.processing_timeout:.1f}s")
        logger.info(f"Frame skip: {self.frame_skip}, Resolution factor: {self.resolution_factor:.2f} ,scene threshold:{self.scene_threshold:.2f}")
        logger.info(f"Keyframe detection: {'Enabled' if self.use_keyframes else 'Disabled'}")
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame based on resolution factor."""
        if self.resolution_factor == 1.0:
            return frame
        
        h, w = frame.shape[:2]
        new_h, new_w = int(h * self.resolution_factor), int(w * self.resolution_factor)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _is_keyframe(self, frame: np.ndarray) -> bool:
        """Detect if the current frame is a keyframe (significant scene change)."""
        if not self.use_keyframes or self.prev_frame is None:
            self.prev_frame = frame.copy()
            return True
        
        # Convert frames to grayscale for comparison
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame similarity using structural similarity index
        score = cv2.matchTemplate(prev_gray, curr_gray, cv2.TM_CCOEFF_NORMED)[0][0]
        is_key = score < (1.0 - self.scene_threshold)
        
        # Update previous frame
        self.prev_frame = frame.copy()
        
        return is_key
    
    async def stream_frames(
        self,
        video_path: str
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Stream video frames asynchronously with timeout protection.
        Applies frame skipping and resolution reduction based on settings.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            Tuple[int, np.ndarray]: Frame number and frame data
        """
        start_time = time.time()
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            frame_count = 0
            frames_processed = 0
            self.prev_frame = None  # Reset previous frame
            
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > self.processing_timeout:
                    logger.warning(
                        f"Video processing timeout reached after {elapsed_time:.1f}s "
                        f"on {self.device} device ({frames_processed}/{frame_count} frames processed)"
                    )
                    break
                
                # Use run_in_executor to prevent blocking the event loop
                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    None, cap.read
                )
                
                if not ret:
                    logger.info(
                        f"Completed processing {frames_processed}/{frame_count} frames "
                        f"in {elapsed_time:.1f}s on {self.device} device"
                    )
                    break
                
                # Process this frame if it's a keyframe or if we're not skipping this frame
                process_this_frame = (
                    (frame_count % (self.frame_skip + 1) == 0) or  # Frame skip logic
                    (self.use_keyframes and self._is_keyframe(frame))  # Keyframe logic
                )
                
                if process_this_frame:
                    # Resize frame if needed
                    resized_frame = self._resize_frame(frame)
                    
                    yield frame_count, resized_frame
                    frames_processed += 1
                
                frame_count += 1
                
                # Small delay to prevent CPU hogging while still processing all frames
                await asyncio.sleep(0)
        
        finally:
            cap.release()
    
    @staticmethod
    def get_video_info(video_path: str) -> sv.VideoInfo:
        """Get video information using supervision."""
        return sv.VideoInfo.from_video_path(video_path)
    
    @staticmethod
    async def ensure_video_readable(video_path: str, timeout: float = 5.0) -> bool:
        """
        Check if video is readable within timeout period.
        
        Args:
            video_path: Path to video file
            timeout: Maximum time to wait for video check
            
        Returns:
            bool: True if video is readable
        """
        try:
            async def _check_video():
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return False
                ret, _ = cap.read()
                cap.release()
                return ret
            
            return await asyncio.wait_for(_check_video(), timeout)
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout while checking video readability: {video_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking video readability: {str(e)}")
            return False 
