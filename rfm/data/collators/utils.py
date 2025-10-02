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


def pad_target_progress(progress_list):
    """Helper function to pad target progress sequences to max length."""
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
    """Convert frames to PIL images if they are numpy arrays or serialized bytes."""
    if frames is None:
        return None

    # If frames are already paths (strings), return as is
    if isinstance(frames, str) or (isinstance(frames, list) and all(isinstance(f, str) for f in frames)):
        return frames

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
        from PIL import Image

        pil_images = []

        # Handle different array shapes
        if len(frames.shape) == 4:  # TxHxWxC
            for i in range(frames.shape[0]):  # Iterate over time dimension
                frame = frames[i]  # HxWxC
                # Convert to PIL Image (already in HxWxC format)
                pil_image = Image.fromarray(frame.astype(np.uint8))
                pil_images.append(pil_image)
        elif len(frames.shape) == 3:  # HxWxC (single frame)
            pil_image = Image.fromarray(frames.astype(np.uint8))
            pil_images.append(pil_image)
        else:
            # Try to reshape as 1D array (backward compatibility)
            print(f"Warning: Unexpected frames shape {frames.shape}, treating as 1D array")
            return frames

        return pil_images

    # If frames are list of numpy arrays, convert each to PIL
    if isinstance(frames, list) and all(isinstance(f, np.ndarray) for f in frames):
        from PIL import Image

        pil_images = []
        for frame in frames:
            # Convert to PIL Image (assuming HxWxC format)
            pil_image = Image.fromarray(frame.astype(np.uint8))
            pil_images.append(pil_image)
        return pil_images

    return frames
