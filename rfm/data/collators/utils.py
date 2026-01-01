import numpy as np
import torch
import av


def write_mp4(frames, out_path, fps=4):
    w, h = frames[0].size
    c = av.open(str(out_path), mode="w")
    s = c.add_stream("libx264", rate=fps)
    s.width, s.height = w, h
    s.pix_fmt = "yuv420p"
    s.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "28",
        "x264-params": "keyint=1:min-keyint=1:scenecut=0",
    }
    for img in frames:
        frame = av.VideoFrame.from_ndarray(np.array(img), format="rgb24")
        for pkt in s.encode(frame):
            c.mux(pkt)
    for pkt in s.encode(None):
        c.mux(pkt)
    c.close()


def pad_list_to_max(progress_list):
    """Helper function to pad lists of sequences to max length."""
    if not progress_list:
        return None

    max_length = max(len(progress) for progress in progress_list)
    padded_list = []
    for progress in progress_list:
        if len(progress) < max_length:
            # Pad with zeros at the end
            padded_progress = progress + [0.0] * (max_length - len(progress))
        else:
            padded_progress = progress
        padded_list.append(padded_progress)
    return torch.tensor(padded_list, dtype=torch.float32)


def convert_frames_to_pil_images(frames, frames_shape=None):
    """Convert frames to PIL images if they are numpy arrays or serialized bytes.

    Handles:
    - Bytes with shape: deserializes to numpy array then converts
    - Numpy arrays (TxHxWxC or HxWxC): converts each frame to PIL Image
    - List of numpy arrays: converts each to PIL Image
    - List of PIL Images: returns as-is
    - List of mixed types (strings, PIL Images, numpy arrays): converts appropriately
    """
    from PIL import Image

    # If frames are serialized bytes, deserialize first
    if isinstance(frames, bytes):
        # Deserialize bytes to numpy array (TxHxWxC) using provided shape
        if frames_shape is not None:
            # Convert to tuple if it's a list
            if isinstance(frames_shape, list):
                frames_shape = tuple(frames_shape)
            try:
                frames = np.frombuffer(frames, dtype=np.uint8).reshape(frames_shape)
            except Exception as e:
                print(f"Warning: Failed to reshape with provided shape {frames_shape}: {e}")
                # Fall back to 1D array
                frames = np.frombuffer(frames, dtype=np.uint8)
        else:
            # No shape provided, try to infer
            frames = np.frombuffer(frames, dtype=np.uint8)

    # If frames are numpy array (TxHxWxC), convert to list of PIL images
    if isinstance(frames, np.ndarray):
        pil_images = []

        # Handle different array shapes
        if len(frames.shape) == 4:  # TxHxWxC
            for i in range(frames.shape[0]):  # Iterate over time dimension
                frame = frames[i]  # HxWxC
                # Ensure uint8 dtype
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)
                pil_images.append(pil_image)
        elif len(frames.shape) == 3:  # HxWxC (single frame)
            frame = frames
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            pil_images.append(pil_image)
        else:
            raise ValueError(f"Unexpected frames shape {frames.shape}. Expected 3D (HxWxC) or 4D (TxHxWxC) array.")

        return pil_images

    # If frames are list, handle each element
    if isinstance(frames, list):
        pil_images = []
        for frame in frames:
            if isinstance(frame, str):
                # File path - open it
                pil_images.append(Image.open(frame))
            elif isinstance(frame, Image.Image):
                # Already PIL Image
                pil_images.append(frame)
            elif isinstance(frame, np.ndarray):
                # Numpy array - convert to PIL
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)
                pil_images.append(pil_image)
            else:
                # Try to convert to numpy array first
                try:
                    frame_array = np.array(frame)
                    if frame_array.dtype != np.uint8:
                        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(frame_array))
                except Exception as e:
                    print(f"Warning: Could not convert frame to PIL Image: {e}")
                    continue
        return pil_images

    raise ValueError(f"Unsupported frames type: {type(frames)}")


def frames_to_numpy_array(frames):
    """Convert frames to a numpy array of shape (T, H, W, C).

    Handles:
    - None: returns None
    - Numpy arrays: returns as-is (ensures uint8 dtype)
    - List of strings (file paths): opens each path and converts to array
    - List of PIL Images: converts each to array
    - List of numpy arrays: stacks them
    """
    from PIL import Image

    if frames is None:
        return None

    if isinstance(frames, np.ndarray):
        # Already numpy array - ensure uint8 dtype
        frames_array = frames
        if frames_array.dtype != np.uint8:
            frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
        return frames_array
    elif isinstance(frames, list):
        # Convert list of images/paths to array
        frame_list = []
        for frame in frames:
            if isinstance(frame, str):
                img = np.array(Image.open(frame))
            elif isinstance(frame, Image.Image):
                img = np.array(frame)
            elif isinstance(frame, np.ndarray):
                img = frame
            else:
                img = np.array(frame)
            frame_list.append(img)
        if frame_list:
            return np.stack(frame_list)

    return None
