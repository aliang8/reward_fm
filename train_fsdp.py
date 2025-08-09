#!/usr/bin/env python3
"""
Clean FSDP Trainer for RFM (Reward Function Model)
Based on the provided FSDP SFT trainer but simplified for RFM use case.
"""

import os
import warnings
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler
from transformers import TrainingArguments
from tqdm import tqdm
from typing import Dict, Any, Optional
import logging
from contextlib import nullcontext

from rfm.data.data_generator import BatchCollator, InfiniteDataGeneratorDataset, DataGenerator
from rfm.models.rfm import RFMModel
from rfm.utils.logging import rank_0_print, is_rank_0
from trainer import RFMTrainer, compute_metrics

# Import shared configs and utilities
from rfm.configs.experiment_configs import ExperimentConfig
from setup_utils import (
    setup_model_and_processor,
    setup_peft_model,
    create_training_arguments,
    setup_data_generator,
    setup_batch_collator,
    setup_train_dataset
)

# Import FSDP utilities
from rfm.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    fsdp2_clip_grad_norm_,
    fsdp_version,
    get_fsdp_state_ctx,
    fsdp2_load_full_state_dict,
    apply_fsdp2,
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard
)

# Import additional utilities
from rfm.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from rfm.utils.distributed import initialize_global_process_group, destroy_global_process_group
from rfm.utils.fsdp_ulysses import FSDPUlyssesShardingManager

# Suppress FSDP warnings
warnings.filterwarnings("ignore", message="Please use DTensor instead and we are deprecating ShardedTensor")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class FSDPTrainer:
    """
    FSDP trainer for RFM model training.
    
    This trainer properly uses rfm.utils.fsdp_utils for:
    - Wrap policy configuration
    - Model initialization
    - Gradient clipping
    - Checkpoint saving/loading
    - FSDP1/FSDP2 compatibility
    """
    
    def __init__(self, model, training_args, train_dataset, data_collator, device_mesh, 
                 ulysses_device_mesh: DeviceMesh = None, fsdp_strategy: str = "fsdp2"):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = None # No validation dataset in this simplified trainer
        self.data_collator = data_collator
        self.training_args = training_args
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.beta = training_args.beta if hasattr(training_args, 'beta') else 0.1
        self.compute_metrics = compute_metrics # This will be passed from train_with_fsdp
        self.fsdp_strategy = fsdp_strategy
        
        # Setup Ulysses sharding manager if sequence parallelism is enabled
        self.sharding_manager = None
        if ulysses_device_mesh is not None:
            self.sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)
            rank_0_print(f"Ulysses sequence parallelism enabled with mesh: {ulysses_device_mesh.shape}")
        
        # Setup FSDP model
        self._setup_fsdp_model()
        
        # Setup dataloaders
        self._setup_dataloaders()
        
        # Setup optimizer and scheduler
        self._setup_optimizer_scheduler()
        
        # Print setup summary after all initialization is complete
        # self._print_setup_summary()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def _setup_fsdp_model(self):
        """Setup FSDP model with proper configuration using fsdp_utils."""
        # Get FSDP strategy early to avoid UnboundLocalError
        fsdp_strategy = self.fsdp_strategy
        
        # Check if using PEFT first
        is_lora = (hasattr(self.model, 'peft_config') or 
                  any('lora' in name.lower() for name, _ in self.model.named_modules()) or
                  any('peft' in name.lower() for name, _ in self.model.named_modules()))
        
        # FSDP configuration with proper dtype handling for PEFT
        # Use float32 for parameters to avoid dtype mismatches with LoRA
        param_dtype = torch.float32 if is_lora else torch.bfloat16
        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32
        )
        
        # Use fsdp_utils for wrap policy - transformer-based wrapping
        # Define transformer layer classes to wrap
        transformer_layers = [
            "Qwen2_5_VLDecoderLayer",  # Language model layers
            "Qwen2_5_VLVisionBlock",   # Vision model layers
            # "Qwen2_5_VLMLP",           # MLP layers
            # "Qwen2_5_VLAttention",     # Attention layers
            # "Qwen2_5_VLVisionAttention"  # Vision attention layers
        ]
        
        fsdp_config = {
            "wrap_policy": {
                "transformer_layer_cls_to_wrap": transformer_layers
            }
        }
        
        # Get the wrap policy using fsdp_utils
        final_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=fsdp_config["wrap_policy"],
            is_lora=is_lora
        )
        
        # Find embedding modules to ignore
        embedding_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                embedding_modules.append(module)
        
        # CPU offload configuration
        cpu_offload = None
        if hasattr(self.training_args, 'fsdp_offload_params') and self.training_args.fsdp_offload_params:
            cpu_offload = CPUOffload(offload_params=True)
        
        # Use fsdp_utils init function
        param_init_fn = init_fn
        
        # Create FSDP model (support both FSDP1 and FSDP2)
        if fsdp_strategy == "fsdp":
            # FSDP1
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=param_init_fn,
                use_orig_params=True,  # Important for embedding layers
                auto_wrap_policy=final_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            # FSDP2
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, 
                reduce_dtype=torch.float32, 
                cast_forward_inputs=True
            )
            
            # Store the full state dict before applying FSDP2
            full_state = self.model.state_dict()
            
            fsdp_kwargs = {
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            
            # Apply FSDP2 using fsdp_utils
            apply_fsdp2(self.model, fsdp_kwargs, fsdp_config)
            
            # Load the full state dict back into the sharded model
            # This ensures proper initialization of distributed parameters
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"FSDP strategy {fsdp_strategy} not implemented")
        
        # # Store info for later logging (after FSDP setup completes successfully)
        # self._fsdp_setup_info = {
        #     'fsdp_strategy': fsdp_strategy,
        #     'param_dtype': param_dtype,
        #     'is_lora': is_lora,
        #     'embedding_count': len(embedding_modules),
        #     'existing_layers': existing_layers,
        #     'wrap_policy': 'transformer_based' if existing_layers else 'size_based_fallback',
        #     'wrap_policy_details': final_wrap_policy.__name__ if hasattr(final_wrap_policy, '__name__') else str(type(final_wrap_policy))
        # }
        
        # # Ensure embedding layers are on the correct device
        # self._ensure_embedding_device_placement()
        
    def _ensure_embedding_device_placement(self):
        """Ensure all embedding layers and model components are on the correct device after FSDP wrapping."""
        target_device = torch.cuda.current_device()
        moved_embeddings = []
        
        # Check FSDP model
        for name, module in self.fsdp_model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.device != target_device:
                    module.to(target_device)
                    moved_embeddings.append(name)
        
        # Check base model
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.device != target_device:
                    module.to(target_device)
                    moved_embeddings.append(name)
        
        # Ensure all buffers (including rotary embeddings) are on the correct device
        moved_buffers = []
        for name, buffer in self.fsdp_model.named_buffers():
            if buffer.device != target_device:
                buffer.data = buffer.data.to(target_device)
                moved_buffers.append(name)
        
        # Also check base model buffers
        for name, buffer in self.model.named_buffers():
            if buffer.device != target_device:
                buffer.data = buffer.data.to(target_device)
                moved_buffers.append(name)
        
        # Store device placement info for later logging
        device_issues = []
        for name, module in self.fsdp_model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.device != target_device:
                    device_issues.append((name, module.weight.device))
        
        # Check for buffer device issues
        buffer_issues = []
        for name, buffer in self.fsdp_model.named_buffers():
            if buffer.device != target_device:
                buffer_issues.append((name, buffer.device))
        
        self._device_placement_info = {
            'moved_embeddings': moved_embeddings,
            'moved_buffers': moved_buffers,
            'device_issues': device_issues,
            'buffer_issues': buffer_issues,
            'target_device': target_device
        }
        
    def _print_setup_summary(self):
        """Print setup summary after all FSDP setup is complete."""
        if not is_rank_0():
            return
            
        rank_0_print("=== FSDP Setup Complete ===")
        
        # Print FSDP info
        if hasattr(self, '_fsdp_setup_info'):
            info = self._fsdp_setup_info
            rank_0_print(f"FSDP Strategy: {info['fsdp_strategy']}")
            rank_0_print(f"Parameter dtype: {info['param_dtype']}")
            rank_0_print(f"LoRA enabled: {info['is_lora']}")
            rank_0_print(f"Embedding modules found: {info['embedding_count']}")
            rank_0_print(f"Wrap policy: {info['wrap_policy']} (min_params: {info['min_wrap_params']:,})")
            if info['existing_layers']:
                rank_0_print(f"Transformer layers: {', '.join(info['existing_layers'])}")
            else:
                rank_0_print("No transformer layers found - using size-based wrapping")
                
        # Print device placement info
        if hasattr(self, '_device_placement_info'):
            info = self._device_placement_info
            if info['moved_embeddings']:
                rank_0_print(f"Moved {len(info['moved_embeddings'])} embedding layers to {info['target_device']}")
            else:
                rank_0_print(f"All embedding layers already on correct device: {info['target_device']}")
                
            if info['device_issues']:
                rank_0_print("⚠️  WARNING: Some embedding layers still on wrong device:")
                for name, device in info['device_issues']:
                    rank_0_print(f"  - {name}: {device}")
            else:
                rank_0_print("✅ All embedding layers correctly placed")
                
        # Print model training info (from setup_utils)
        if hasattr(self.model, '_training_info'):
            info = self.model._training_info
            rank_0_print(f"Trainable params: {info['trainable_params']:,} || all params: {info['all_params']:,} || trainable%: {100 * info['trainable_params'] / info['all_params']:.4f}")
            rank_0_print("Training configuration:")
            rank_0_print(f"  - Vision encoder: {info['config'].train_vision_encoder}")
            rank_0_print(f"  - Language model: {info['config'].train_language_model}")
            rank_0_print(f"  - Progress head: {info['config'].train_progress_head}")
            rank_0_print(f"  - Preference head: {info['config'].train_preference_head}")
            rank_0_print(f"  - Similarity head: {info['config'].train_similarity_head}")
            
        # Print optimizer info
        if hasattr(self, '_optimizer_info'):
            info = self._optimizer_info
            rank_0_print(f"Optimizer and scheduler created - Total steps: {info['total_steps']}")
            
        # Print dataloader info
        if hasattr(self, 'train_dataloader'):
            rank_0_print(f"DataLoaders created - Train: {len(self.train_dataloader)} batches")
        
        rank_0_print("=== End Setup Summary ===")

    def _setup_dataloaders(self):
        """Setup distributed dataloaders."""
        # Training dataloader
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=self.device_mesh.size(),
            rank=self.device_mesh.get_rank(),
            drop_last=True
        )
        
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            sampler=self.train_sampler,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.training_args.dataloader_pin_memory,
            drop_last=True,
            collate_fn=self.data_collator,
        )
        
        # Validation dataloader (if provided)
        if self.val_dataset is not None:
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                shuffle=False,
                num_replicas=self.device_mesh.size(),
                rank=self.device_mesh.get_rank(),
                drop_last=True
            )
            
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.training_args.per_device_eval_batch_size,
                sampler=self.val_sampler,
                num_workers=self.training_args.dataloader_num_workers,
                pin_memory=self.training_args.dataloader_pin_memory,
                drop_last=True,
                collate_fn=self.data_collator,
            )
        
        # _print_setup_summary() # Moved outside to ensure all setup is complete
        
    def _setup_optimizer_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.training_args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_dataloader) * self.training_args.num_train_epochs
        warmup_steps = int(total_steps * 0.1)  # 10% warmup
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 4,  # Restart every quarter
            T_mult=1,
            eta_min=self.training_args.learning_rate * 0.1
        )
        
        # Store for summary
        self._optimizer_info = {'total_steps': total_steps, 'warmup_steps': warmup_steps}
        
    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss using the RFM trainer's logic with optional Ulysses support."""
        # Use Ulysses sharding manager if available
        context = self.sharding_manager if self.sharding_manager is not None else nullcontext()
        
        with context:
            # Create a temporary RFM trainer to use its compute_loss method
            temp_trainer = RFMTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                data_collator=self.data_collator,
                beta=self.beta,
                compute_metrics=self.compute_metrics,
            )
            
            # Compute loss
            loss = temp_trainer.compute_loss(self.fsdp_model, batch)
            return loss
        
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        self.fsdp_model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping using fsdp_utils
        if hasattr(self.training_args, 'max_grad_norm') and self.training_args.max_grad_norm > 0:
            if fsdp_version(self.fsdp_model) == 2:
                grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.training_args.max_grad_norm)
            else:
                grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.training_args.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # Get current learning rate
        current_lr = self.lr_scheduler.get_last_lr()[0]
        
        return {
            "loss": loss.item(),
            "learning_rate": current_lr,
        }
        
    def validation_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Single validation step."""
        self.fsdp_model.eval()
        
        with torch.no_grad():
            loss = self.compute_loss(batch)
            
        return loss
        
    def save_checkpoint(self, step: int, output_dir: str):
        """Save model checkpoint using fsdp_utils."""
        rank_0_print(f"Saving checkpoint at step {step}...")
        
        # Create output directory
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        if is_rank_0():
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save FSDP model using fsdp_utils
        if fsdp_version(self.fsdp_model) == 1:
            # FSDP1 checkpoint saving
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with get_fsdp_state_ctx(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg, None):
                state_dict = self.fsdp_model.state_dict()
        else:
            # FSDP2 checkpoint saving
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)
            
        if is_rank_0():
            # Save model
            self.model.save_pretrained(checkpoint_dir, state_dict=state_dict)
            
            # Save training arguments
            self.training_args.save_to_json(os.path.join(checkpoint_dir, "training_args.json"))
            
            rank_0_print(f"Checkpoint saved to {checkpoint_dir}")
            
        # Synchronize all processes
        dist.barrier()
        
    def train(self, max_steps: Optional[int] = None):
        """Main training loop."""
        rank_0_print("Starting training...")
        
        # Training loop
        for epoch in range(self.training_args.num_train_epochs):
            self.epoch = epoch
            self.train_sampler.set_epoch(epoch)
            
            # Training epoch
            train_losses = []
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.training_args.num_train_epochs}",
                disable=not is_rank_0()
            )
            
            for batch in progress_bar:
                self.global_step += 1
                
                # Training step
                metrics = self.training_step(batch)
                train_losses.append(metrics["loss"])
                
                # Update progress bar
                if is_rank_0():
                    progress_bar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "lr": f"{metrics['learning_rate']:.2e}"
                    })
                
                # Logging
                if self.global_step % self.training_args.logging_steps == 0:
                    avg_loss = sum(train_losses[-self.training_args.logging_steps:]) / len(train_losses[-self.training_args.logging_steps:])
                    rank_0_print(f"Step {self.global_step}: Loss = {avg_loss:.4f}, LR = {metrics['learning_rate']:.2e}")
                
                # Save checkpoint
                if self.global_step % self.training_args.save_steps == 0:
                    self.save_checkpoint(self.global_step, self.training_args.output_dir)
                
                # Validation
                if (self.val_dataset is not None and 
                    self.global_step % self.training_args.eval_steps == 0):
                    self.evaluate()
                
                # Check if we've reached max steps
                if max_steps is not None and self.global_step >= max_steps:
                    rank_0_print(f"Reached max steps ({max_steps}), stopping training")
                    break
            
            # End of epoch validation
            if self.val_dataset is not None:
                self.evaluate()
                
        rank_0_print("Training completed!")
        
    def evaluate(self):
        """Run evaluation."""
        if self.val_dataset is None:
            return
            
        rank_0_print("Running evaluation...")
        
        self.fsdp_model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", disable=not is_rank_0()):
                loss = self.validation_step(batch)
                val_losses.append(loss.item())
        
        # Average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        rank_0_print(f"Validation Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss


def setup_fsdp_environment():
    """Setup FSDP environment using proper utilities."""
    # Set environment variables
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize distributed training using proper utilities
    # This now handles duplicate initialization gracefully
    local_rank, rank, world_size = initialize_global_process_group()
    
    # Create device mesh using proper device detection with named dimensions
    device_name = get_device_name()
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    
    rank_0_print(f"FSDP environment initialized: {world_size} devices, rank {rank}, device={device_name}")
    
    return local_rank, rank, world_size, device_mesh


def train_with_fsdp(
    cfg: ExperimentConfig,
    ulysses_sequence_parallel_size: int = 1,
    fsdp_strategy: str = None  # Will default to config value if None
):
    """
    Main function to run FSDP training.
    
    Args:
        cfg: Experiment configuration
        ulysses_sequence_parallel_size: Size for Ulysses sequence parallelism (disabled for now)
        fsdp_strategy: FSDP strategy override, defaults to cfg.training.fsdp_strategy
    """
    # Use config value if not explicitly provided
    if fsdp_strategy is None:
        fsdp_strategy = getattr(cfg.training, 'fsdp_strategy', 'fsdp2')
    
    # Setup FSDP environment
    local_rank, rank, world_size, device_mesh = setup_fsdp_environment()
    
    # Set up model, processor, data generator, and datasets from config
    processor, model = setup_model_and_processor(cfg)
    model = setup_peft_model(model, cfg)
    
    # Create data generator and datasets
    data_generator = setup_data_generator(cfg)
    batch_collator = setup_batch_collator(processor, cfg)
    train_dataset = setup_train_dataset(data_generator)
    
    # Create training arguments
    training_args = create_training_arguments(cfg, cfg.training.output_dir)
    
    # Handle Ulysses sequence parallelism setup (currently disabled)
    ulysses_device_mesh = None
    if ulysses_sequence_parallel_size > 1:
        # Temporarily disable Ulysses sequence parallelism due to device mesh conflicts
        rank_0_print(f"⚠️  WARNING: Ulysses sequence parallelism disabled due to device mesh conflicts")
        rank_0_print(f"   Requested SP size: {ulysses_sequence_parallel_size}")
        rank_0_print(f"   Using FSDP-only training for now")
        ulysses_sequence_parallel_size = 1
    
    # Create trainer
    trainer = FSDPTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=batch_collator,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        fsdp_strategy=fsdp_strategy,
    )
    
    # Start training
    trainer.train()
    
    # Cleanup using proper utility
    destroy_global_process_group()
    
    return trainer


if __name__ == "__main__":
    """Standalone script to run FSDP training."""
    from pyrallis import wrap
    
    @wrap()
    def main(cfg: ExperimentConfig):
        """Main function for FSDP training with pyrallis config management."""
        from rich.console import Console
        from rich.panel import Panel
        from rich import print as rprint
        
        # Display the configuration in a nice Rich format
        if is_rank_0():
            console = Console()
            console.print(Panel.fit("🚀 Starting RFM FSDP Training", style="bold green"))
            
            # Create a table for the main config
            from rich.table import Table
            table = Table(title="FSDP Training Settings", show_header=True, header_style="bold magenta")
            table.add_column("Section", style="cyan", no_wrap=True)
            table.add_column("Key", style="green")
            table.add_column("Value", style="yellow")
            
            # Model config
            table.add_row("Model", "Base Model", cfg.model.base_model_id)
            table.add_row("Model", "Torch Dtype", cfg.model.torch_dtype)
            table.add_row("Model", "Trust Remote Code", str(cfg.model.trust_remote_code))
            
            # Data config
            table.add_row("Data", "Dataset Path", cfg.data.dataset_path)
            table.add_row("Data", "Dataset Subsets", ", ".join(cfg.data.dataset_subsets))
            table.add_row("Data", "Max Frames", str(cfg.data.max_frames))
            table.add_row("Data", "Resized Height", str(cfg.data.resized_height))
            table.add_row("Data", "Resized Width", str(cfg.data.resized_width))
            table.add_row("Data", "Preference Ratio", str(cfg.data.preference_ratio))
            table.add_row("Data", "Similarity Ratio", str(cfg.data.similarity_ratio))
            
            # Training config
            table.add_row("Training", "Number of GPUs", str(cfg.training.num_gpus))
            table.add_row("Training", "Output Directory", cfg.training.output_dir)
            table.add_row("Training", "Batch Size", str(cfg.training.per_device_train_batch_size))
            table.add_row("Training", "Learning Rate", f"{cfg.training.learning_rate:.2e}")
            table.add_row("Training", "Epochs", str(cfg.training.num_train_epochs))
            table.add_row("Training", "Max Seq Length", str(cfg.training.max_seq_length))
            table.add_row("Training", "Beta", str(cfg.training.beta))
            table.add_row("Training", "Logging Steps", str(cfg.training.logging_steps))
            table.add_row("Training", "Save Steps", str(cfg.training.save_steps))
            
            # PEFT config
            table.add_row("PEFT", "Use PEFT", str(cfg.peft.use_peft))
            if cfg.peft.use_peft:
                table.add_row("PEFT", "LoRA Rank (r)", str(cfg.peft.r))
                table.add_row("PEFT", "LoRA Alpha", str(cfg.peft.lora_alpha))
                table.add_row("PEFT", "Target Modules", ", ".join(cfg.peft.target_modules))
            
            console.print(table)
        
        # Setup model and processor using shared utilities
        rank_0_print("Setting up model and processor...")
        processor, rfm_model = setup_model_and_processor(cfg)
        
        # Apply PEFT if requested using shared utilities
        if cfg.peft.use_peft:
            rfm_model = setup_peft_model(rfm_model, cfg)
        
        # Create data generator using shared utilities
        rank_0_print("Setting up data generator...")
        data_generator = setup_data_generator(cfg)
        
        # Create dataset and batch collator using shared utilities
        train_dataset = setup_train_dataset(data_generator)
        batch_collator = setup_batch_collator(processor, cfg)
        
        # Create training arguments using shared utilities
        training_args = create_training_arguments(cfg, cfg.training.output_dir)
        
        # Start training
        rank_0_print("Starting FSDP training...")
        trainer = train_with_fsdp(
            cfg=cfg,
            ulysses_sequence_parallel_size=getattr(cfg.training, 'ulysses_sequence_parallel_size', 1),
            fsdp_strategy=getattr(cfg.training, 'fsdp_strategy', 'fsdp2'),
        )
        
        rank_0_print("Training completed!")
    
    main() 