import os
import json
from typing import Dict, Any, Iterable, Optional, List
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(
        self,
        log_to: Iterable[str] | None,
        output_dir: str,
        is_main_process: bool = True,
        wandb_run: Optional[Any] = None,
    ):
        backends = [b.lower() for b in (list(log_to) if log_to is not None else [])]
        self._use_wandb = "wandb" in backends
        self._use_tb = "tensorboard" in backends
        self._is_main = bool(is_main_process)

        self._wandb_run = wandb.run if (self._use_wandb and self._is_main) else None

        self._tb_writer = None
        if self._use_tb and self._is_main:
            logging_dir = os.path.join(output_dir, "tb")
            os.makedirs(logging_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=logging_dir)

    def init_wandb(self, project: Optional[str], entity: Optional[str], name: Optional[str], config: Optional[dict]):
        if not (self._use_wandb and self._is_main):
            return None
        if self._wandb_run is not None:
            return self._wandb_run
        self._wandb_run = wandb.init(project=project, entity=entity, name=name, config=config)
        return self._wandb_run

    def enabled(self, backend: str) -> bool:
        backend = backend.lower()
        if backend == "wandb":
            return self._use_wandb and self._is_main and (self._wandb_run is not None)
        if backend == "tensorboard":
            return self._use_tb and self._is_main and self._tb_writer is not None
        return False

    def log_scalars(self, scalars: Dict[str, float], step: Optional[int] = None):
        if not self._is_main:
            return
        if self.enabled("wandb"):
            self._wandb_run.log(scalars, step=step)
        if self.enabled("tensorboard"):
            for k, v in scalars.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, float(v), global_step=step)

    def log_figure(self, tag: str, figure, step: Optional[int] = None):
        if not self._is_main:
            return
        if self.enabled("wandb"):
            self._wandb_run.log({tag: wandb.Image(figure)}, step=step)
        if self.enabled("tensorboard"):
            self._tb_writer.add_figure(tag, figure, global_step=step)

    def log_video_table(
        self, tag: str, videos_and_figures: List[tuple], columns: List[str], step: Optional[int] = None
    ):
        """
        Log a table where first column can be video (wandb), second a figure, etc.
        videos_and_figures: list of tuples e.g. [(video_array_or_path, figure), ...]
        Only supported for wandb; TensorBoard has no native table/video support.
        """
        if not self._is_main:
            return
        if self.enabled("wandb"):
            rows = []
            for item in videos_and_figures:
                row = []
                for x in item:
                    if x is None:
                        row.append(None)
                    else:
                        # Heuristically wrap images/figures; assume pre-encoded videos passed as wandb.Video externally
                        if hasattr(x, "savefig") or getattr(x, "__class__", type("x", (), {})).__name__ == "Figure":
                            row.append(wandb.Image(x))
                        else:
                            row.append(x)
                rows.append(row)
            self._wandb_run.log({tag: wandb.Table(data=rows, columns=columns)}, step=step)

    def add_text(self, tag: str, text: str, step: Optional[int] = None):
        if not self._is_main:
            return
        if self.enabled("tensorboard"):
            self._tb_writer.add_text(tag, text, global_step=step)
        # For wandb, text can be added via log_scalars or a media panel; skip for simplicity
        if self.enabled("wandb"):
            # Store as a simple text panel by wrapping in a dict
            self._wandb_run.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def log_table(self, tag: str, data: List[List[Any]], columns: List[str], step: Optional[int] = None):
        """
        Log a generic table (wandb only). TensorBoard has no native table support.
        """
        if not self._is_main:
            return
        if self.enabled("wandb"):
            self._wandb_run.log({tag: wandb.Table(data=data, columns=columns)}, step=step)

    def log_video(self, tag: str, video: Any, fps: int = 10, step: Optional[int] = None):
        """
        Log a single video clip.
        - For wandb: accepts file path or numpy/torch array; arrays are expected as T x C x H x W or T x H x W x C.
        - For TensorBoard: accepts numpy/torch array; converted to 1 x C x T x H x W with values in [0,1].
        """
        if not self._is_main:
            return
        # wandb
        if self.enabled("wandb"):
            if isinstance(video, str):
                self._wandb_run.log({tag: wandb.Video(video, fps=fps)})
            else:
                arr = None
                if isinstance(video, np.ndarray):
                    arr = video
                elif isinstance(video, torch.Tensor):
                    arr = video.detach().cpu().numpy()
                if arr is not None and arr.ndim == 4:
                    # Convert T x H x W x C -> T x C x H x W if needed
                    if arr.shape[-1] in (1, 3):
                        arr = np.transpose(arr, (0, 3, 1, 2))
                    self._wandb_run.log({tag: wandb.Video(arr, fps=fps, format="mp4")}, step=step)
        # tensorboard
        if self.enabled("tensorboard"):
            tens = None
            if isinstance(video, torch.Tensor):
                tens = video.detach().cpu()
            elif isinstance(video, np.ndarray):
                tens = torch.from_numpy(video)
            if tens is not None and tens.dim() == 4:
                # Convert to C x T x H x W
                if tens.shape[-1] in (1, 3):  # T x H x W x C
                    tens = tens.permute(0, 3, 1, 2)  # T x C x H x W
                if tens.dtype != torch.float32:
                    tens = tens.float()
                if tens.numel() > 0 and tens.max().item() > 1.0:
                    tens = tens / 255.0
                tens = tens.permute(1, 0, 2, 3).unsqueeze(0)  # 1 x C x T x H x W
                self._tb_writer.add_video(tag, tens, global_step=step, fps=fps)

    def close(self):
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()

    def write_wandb_info(self, output_dir: str, run_name: str):
        """
        Persist basic wandb run information alongside outputs, if wandb is active.
        Safe to call even if wandb isn't enabled or initialized.
        """
        if not self.enabled("wandb"):
            return
        run = self._wandb_run
        if run is None:
            return
        info = {
            "wandb_id": run.id,
            "wandb_name": run.name or run_name,
            "wandb_project": run.project,
            "wandb_entity": run.entity,
            "wandb_url": run.url,
        }
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "wandb_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
