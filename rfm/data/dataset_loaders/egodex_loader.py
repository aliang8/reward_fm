"""
EgoDex dataset loader for the generic dataset converter for RFM model training.
This module contains EgoDex-specific logic for loading and processing data iteratively
to handle large datasets without loading everything into memory at once.
"""

import os
import h5py
import numpy as np
import cv2
from typing import Iterator, Dict, Tuple, Optional, List
from pathlib import Path
from tqdm import tqdm


class EgoDexIterator:
    """Iterator for EgoDex dataset that yields trajectories one at a time to manage memory efficiently."""
    
    def __init__(self, dataset_path: str, max_trajectories: Optional[int] = None):
        """
        Initialize EgoDex iterator.
        
        Args:
            dataset_path: Path to the EgoDex dataset directory
            max_trajectories: Maximum number of trajectories to process (None for all)
        """
        self.dataset_path = Path(os.path.expanduser(dataset_path))
        self.max_trajectories = max_trajectories
        self.trajectory_count = 0
        
        # Validate dataset path
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Find all trajectory files
        self.trajectory_files = self._discover_trajectory_files()
        
        print(f"Found {len(self.trajectory_files)} trajectory pairs")
        if max_trajectories:
            print(f"Will process up to {max_trajectories} trajectories")
    
    def _discover_trajectory_files(self) -> List[Tuple[Path, Path, str]]:
        """
        Discover all trajectory file pairs (HDF5 + MP4) in the dataset.
        
        Returns:
            List of tuples (hdf5_path, mp4_path, task_name)
        """
        trajectory_files = []
        
        # Dataset path directly contains task directories
        for task_dir in self.dataset_path.iterdir():
            if not task_dir.is_dir():
                continue
            
            task_name = task_dir.name
            
            # Find all HDF5 files in this task directory
            for hdf5_file in task_dir.glob("*.hdf5"):
                # Check if corresponding MP4 file exists
                mp4_file = hdf5_file.with_suffix('.mp4')
                if mp4_file.exists():
                    trajectory_files.append((hdf5_file, mp4_file, task_name))
                else:
                    print(f"Warning: Missing MP4 file for {hdf5_file}")
        
        return trajectory_files

    
    def __iter__(self):
        """Return iterator."""
        self.trajectory_count = 0
        return self
    
    def __next__(self) -> Dict:
        """
        Get next trajectory.
        
        Returns:
            Dictionary containing trajectory data
        """
        if self.max_trajectories and self.trajectory_count >= self.max_trajectories and self.max_trajectories != -1:
            raise StopIteration
        
        if self.trajectory_count >= len(self.trajectory_files) and self.max_trajectories != -1:
            raise StopIteration
        
        hdf5_path, mp4_path, task_name = self.trajectory_files[self.trajectory_count]
        
        try:
            # Load trajectory data
            trajectory = self._load_trajectory(hdf5_path, mp4_path, task_name)
            self.trajectory_count += 1
            return trajectory
        except Exception as e:
            print(f"Error loading trajectory {hdf5_path}: {e}")
            self.trajectory_count += 1
            # Skip this trajectory and try the next one
            return self.__next__()
    
    def _load_trajectory(self, hdf5_path: Path, mp4_path: Path, task_name: str) -> Dict:
        """
        Load trajectory metadata with a callable for lazy frame loading.
        
        Args:
            hdf5_path: Path to HDF5 file with pose annotations
            mp4_path: Path to MP4 video file
            task_name: Name of the task
            
        Returns:
            Dictionary containing trajectory metadata and frame loader function
        """
        # Load pose data and metadata from HDF5 (lightweight)
        pose_data, task_description = self._load_hdf5_data(hdf5_path)
        
        # Create a frame loader function that will be called on-demand
        def load_frames():
            """Load frames from the MP4 file when called."""
            return self._load_video_frames(mp4_path)
        
        # Create trajectory dictionary with frame loader function
        trajectory = {
            'frames': load_frames,  # Callable that returns frames when needed
            'actions': pose_data,  # Use pose data as actions
            'is_robot': False,  # EgoDex is human egocentric data
            'task': task_description or f"EgoDex {task_name}",
            'optimal': 'optimal',  # Assume demonstrations are optimal
            'task_name': task_name,
            'trajectory_id': f"{task_name}_{hdf5_path.stem}"
        }
        
        return trajectory
    

    
    def _load_video_frames(self, mp4_path: Path) -> np.ndarray:
        """
        Load video frames from MP4 file with robust codec handling.
        
        Args:
            mp4_path: Path to MP4 video file
            
        Returns:
            Numpy array of video frames (T, H, W, C)
        """
        # Try multiple backends for better codec support
        backends = [
            cv2.CAP_FFMPEG,  # FFmpeg backend (best for various codecs)
            cv2.CAP_ANY,     # Any available backend
        ]
        
        cap = None
        for backend in backends:
            try:
                cap = cv2.VideoCapture(str(mp4_path), backend)
                if cap.isOpened():
                    # Test if we can actually read frames
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        # Reset to beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    if cap:
                        cap.release()
                    cap = None
            except Exception as e:
                print(f"Backend {backend} failed for {mp4_path}: {e}")
                if cap:
                    cap.release()
                cap = None
        
        if cap is None or not cap.isOpened():
            raise ValueError(f"Could not open video file with any backend: {mp4_path}")
        
        frames = []
        frame_count = 0
        max_frames_to_try = 10000  # Prevent infinite loops
        
        while frame_count < max_frames_to_try:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Convert BGR to RGB for consistency
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                frame_count += 1
            except Exception as e:
                print(f"Warning: Failed to convert frame {frame_count} from {mp4_path}: {e}")
                continue
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames could be extracted from video: {mp4_path}")
        
        print(f"  📹 Loaded {len(frames)} frames from {mp4_path.name}")
        return np.array(frames)
    
    def _load_hdf5_data(self, hdf5_path: Path) -> Tuple[np.ndarray, str]:
        """
        Load pose data and metadata from HDF5 file.
        
        Args:
            hdf5_path: Path to HDF5 file
            
        Returns:
            Tuple of (pose_data, task_description)
        """
        with h5py.File(hdf5_path, 'r') as f:
            # Extract task description from attributes
            task_description = ""
            try:
                if 'llm_description' in f.attrs:
                    # Check which description to use for reversible tasks
                    if 'which_llm_description' in f.attrs:
                        which_desc = f.attrs['which_llm_description']
                        if which_desc == 2 and 'llm_description2' in f.attrs:
                            desc = f.attrs['llm_description2']
                        else:
                            desc = f.attrs['llm_description']
                    else:
                        desc = f.attrs['llm_description']
                    
                    # Handle both string and bytes
                    if isinstance(desc, bytes):
                        task_description = desc.decode('utf-8')
                    else:
                        task_description = str(desc)
            except Exception as e:
                print(f"Warning: Could not load task description from {hdf5_path}: {e}")
            
            # Extract pose data - use hand positions as primary actions
            pose_data = self._extract_pose_actions(f)
        
        return pose_data, task_description
    
    def _extract_pose_actions(self, hdf5_file) -> np.ndarray:
        """
        Extract pose data to use as actions.
        
        Args:
            hdf5_file: Open HDF5 file handle
            
        Returns:
            Numpy array of pose actions (T, D) where D is action dimension
        """
        actions = []
        
        # Priority order for extracting pose data
        pose_keys = [
            'transforms/leftHand',
            'transforms/rightHand', 
            'transforms/leftIndexFingerTip',
            'transforms/rightIndexFingerTip',
            'transforms/camera'
        ]
        
        for key in pose_keys:
            if key in hdf5_file:
                transform_data = hdf5_file[key][:]  # Shape: (N, 4, 4)
                
                # Extract position (translation) from transformation matrices
                positions = transform_data[:, :3, 3]  # Shape: (N, 3)
                actions.append(positions)
        
        if not actions:
            # Fallback: use camera transform if nothing else available
            if 'transforms/camera' in hdf5_file:
                camera_transforms = hdf5_file['transforms/camera'][:]
                camera_positions = camera_transforms[:, :3, 3]
                actions.append(camera_positions)
            else:
                # Last resort: create dummy actions
                print(f"Warning: No pose data found, creating dummy actions")
                # Assume some reasonable number of frames (will be corrected by video length)
                num_frames = 100
                actions.append(np.zeros((num_frames, 3)))
        
        # Concatenate all action components
        concatenated_actions = np.concatenate(actions, axis=1)  # Shape: (N, total_dim)
        
        return concatenated_actions

    def __len__(self):
        """Return the number of trajectories."""
        return len(self.trajectory_files)


def load_egodex_dataset(dataset_path: str, max_trajectories: int = 100) -> Dict[str, List[Dict]]:
    """
    Load EgoDex dataset using iterator for memory efficiency.
    
    Args:
        dataset_path: Path to the EgoDex dataset directory
        max_trajectories: Maximum number of trajectories to process

        
    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """
    print(f"Loading EgoDex dataset from: {dataset_path}")
    print("=" * 100)
    print("LOADING EGODEX DATASET")
    print("=" * 100)
    
    # Create iterator
    egodex_iterator = EgoDexIterator(dataset_path, max_trajectories)
    
    # Group trajectories by task
    task_data = {}
    
    print(f"Processing trajectories iteratively...")
    for trajectory in tqdm(egodex_iterator, desc="Loading trajectories", total=min(max_trajectories or float('inf'), len(egodex_iterator.trajectory_files))):
        task_name = trajectory['task_name']
        
        if task_name not in task_data:
            task_data[task_name] = []
        
        task_data[task_name].append(trajectory)
        
        # Print progress every 10 trajectories
        if len(task_data) > 0 and sum(len(trajs) for trajs in task_data.values()) % 10 == 0:
            total_loaded = sum(len(trajs) for trajs in task_data.values())
            print(f"  📊 Loaded {total_loaded} trajectories from {len(task_data)} tasks")
    
    total_trajectories = sum(len(trajectories) for trajectories in task_data.values())
    print(f"Loaded {total_trajectories} trajectories from {len(task_data)} tasks")
    
    return task_data


def get_egodex_iterator(dataset_path: str, max_trajectories: Optional[int] = None) -> EgoDexIterator:
    """
    Get an EgoDex iterator for streaming processing.
    
    Args:
        dataset_path: Path to the EgoDex dataset directory
        max_trajectories: Maximum number of trajectories to process

        
    Returns:
        EgoDexIterator instance
    """
    return EgoDexIterator(dataset_path, max_trajectories)