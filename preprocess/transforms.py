import torch
import torchvision
import cv2
import os
from typing import Optional


class VideoFolderPathToTensor(object):
    """Convert a folder of video frames to a normalized tensor.
    
    Args:
        max_len: Maximum number of frames to process (not currently used)
    """
    def __init__(self, max_len: Optional[int] = None):
        self.max_len = max_len
        self.num_time_steps = 16
        self.extract_frequency = 1
        self.target_size = (224, 224)
        
        # Predefined ImageNet normalization parameters
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(self.target_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalize_mean, self.normalize_std),
        ])

    def __call__(self, path: str) -> Optional[torch.Tensor]:
        """Convert frames in directory to normalized tensor.
        
        Args:
            path: Directory containing video frames
            
        Returns:
            Tensor of shape (3, num_time_steps, height, width) or None if error occurs
        """
        try:
            # Get sorted list of frame files
            frame_files = sorted([
                os.path.join(path, f) 
                for f in os.listdir(path) 
                if os.path.isfile(os.path.join(path, f))
            ])
            
            if not frame_files:
                raise ValueError(f"No valid frame files found in {path}")

            required_frames = self.num_time_steps * self.extract_frequency
            if len(frame_files) < required_frames:
                raise ValueError(
                    f"Insufficient frames: {len(frame_files)} < {required_frames} "
                    f"(needed {required_frames} frames)"
                )

            # Initialize output tensor
            frames_tensor = torch.zeros(
                3,  # channels
                self.num_time_steps,
                *self.target_size,
                dtype=torch.float32
            )

            for idx in range(self.num_time_steps):
                frame_idx = idx * self.extract_frequency
                if frame_idx >= len(frame_files):
                    raise ValueError(
                        f"Frame index {frame_idx} out of range "
                        f"(total frames: {len(frame_files)})"
                    )

                frame = self._load_and_validate_frame(frame_files[frame_idx])
                frames_tensor[:, idx, :, :] = self.transform(frame)

            return frames_tensor

        except Exception as e:
            print(f"Error processing video frames from {path}: {str(e)}")
            return None

    def _load_and_validate_frame(self, frame_path: str) -> torch.Tensor:
        """Load and validate a single video frame.
        
        Args:
            frame_path: Path to the frame image file
            
        Returns:
            Tensor of the frame in CHW format
            
        Raises:
            ValueError: If frame is invalid
        """
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to read frame at {frame_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Invalid frame dimensions {frame.shape} "
                f"(expected HxWx3 with 3 channels)"
            )

        return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0