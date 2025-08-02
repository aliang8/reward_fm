# Your main script (e.g., train_dpo.py) - CORRECTED

from re import M
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from PIL import Image
import json
import os
import yaml
from typing import List, Dict, Optional, Union, Any
from peft import get_peft_model, LoraConfig, PeftModel
from data.data_generator import BatchCollator, DataGeneratorDataset
from models.rfm import RFMModel
from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from pyrallis import wrap
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator
import wandb

# Suppress FSDP ShardedTensor deprecation warning
warnings.filterwarnings("ignore", message="Please use DTensor instead and we are deprecating ShardedTensor")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ModelConfig:
    """Config for model settings"""
    base_model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    torch_dtype: str = field(default="bfloat16")
    trust_remote_code: bool = field(default=True)


@dataclass
class PEFTConfig:
    """Config for PEFT/LoRA settings"""
    use_peft: bool = field(default=True, metadata={"help": "Whether to use PEFT/LoRA or train full model"})
    r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    bias: str = field(default="none")
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # Additional options for more comprehensive training
    train_vision_encoder: bool = field(default=False, metadata={"help": "Whether to train the vision encoder"})
    train_language_model: bool = field(default=True, metadata={"help": "Whether to train the language model"})
    train_value_head: bool = field(default=True, metadata={"help": "Whether to train the value head"})


@dataclass
class TrainingConfig:
    """Config for training settings"""
    dpo_dataset_path: str = field(default="rfm_dataset/libero")
    base_dir: str = field(default="libero_dpo_dataset")
    output_dir: str = field(default="./rfm_model_output")
    max_seq_length: int = field(default=1024)
    beta: float = field(default=0.1)
    resume_from_checkpoint: Optional[str] = field(default=None)
    
    # Training arguments
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=5e-7)
    num_train_epochs: Optional[int] = field(default=1)  # Default to 1 epoch if not specified
    save_strategy: str = field(default="steps")
    logging_steps: int = field(default=10)
    bf16: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    fsdp: str = field(default="full_shard")  # Re-enable FSDP
    dataloader_pin_memory: bool = field(default=False)
    dataloader_num_workers: int = field(default=0)
    ddp_find_unused_parameters: bool = field(default=False)
    ddp_bucket_cap_mb: int = field(default=25)
    max_steps: Optional[int] = field(default=-1)  # -1 means no limit, use num_train_epochs instead
    save_steps: int = field(default=100)
    
    # Video processing settings
    video_frame_sampling: str = field(default="uniform")  # "uniform", "random", "first", "middle"
    video_max_frames: int = field(default=8)  # LIBERO dataset uses 8 frames per trajectory
    
    # FSDP configuration (nested under training in YAML)
    fsdp_config: Dict = field(default_factory=lambda: {
        "fsdp_transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],
        "fsdp_offload_params": True
    })

@dataclass
class LoggingConfig:
    """Config for logging settings"""
    print_trainable_parameters: bool = field(default=True)
    save_model: bool = field(default=True)
    save_processor: bool = field(default=True)
    # Wandb configuration
    use_wandb: bool = field(default=True, metadata={"help": "Whether to use Weights & Biases logging"})
    wandb_project: str = field(default="rfm-model", metadata={"help": "Wandb project name"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Wandb entity/username"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "Wandb run name"})


@dataclass
class PromptConfig:
    """Config for prompt settings"""
    discriminator: str = field(default="You are shown three video sequences of robot trajectories. Sequences A and B are from the same task, while sequence C is from a different task. Which sequence (A or B) shows a better trajectory for the task?")


@dataclass
class EvaluationConfig:
    """Config for evaluation settings"""
    model_path: str = field(default="./rfm_model_output")
    eval_subset_size: int = field(default=10, metadata={"help": "Number of examples to use for evaluation"})
    eval_dataset_path: str = field(default="rfm_dataset/libero")
    eval_base_dir: str = field(default="libero_dpo_dataset")


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    mode: str = field(default="train", metadata={"help": "Mode: 'train' or 'evaluate'"})
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
        
class RFMTrainer(Trainer):
    def __init__(self, *args, beta=0.1, compute_metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.compute_metrics = compute_metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Determine prediction type from inputs
        prediction_type = inputs.get("prediction_type", "similarity")  # Default to similarity
        
        if prediction_type == "progress":
            return self._compute_progress_loss(model, inputs, return_outputs)
        elif prediction_type == "preference":
            return self._compute_preference_loss(model, inputs, return_outputs)
        else:  # similarity
            return self._compute_similarity_loss(model, inputs, return_outputs)
    
    def _compute_progress_loss(self, model, inputs, return_outputs=False):
        """Compute progress prediction loss (MSE for frame progress 0-1)."""
        # Forward pass with progress prediction
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            prediction_type="progress"
        )
        
        # Get predicted progress scores
        predicted_progress = outputs.logits.squeeze(-1)
        
        # Get target progress (frame index / total frames)
        target_progress = inputs["target_progress"]  # Should be provided in inputs
        
        # Compute MSE loss
        loss = F.mse_loss(predicted_progress, target_progress)
        
        if return_outputs:
            return loss, {"predicted_progress": predicted_progress, "target_progress": target_progress}
        return loss
    
    def _compute_preference_loss(self, model, inputs, return_outputs=False):
        """Compute preference prediction loss using Bradley-Terry model."""
        # Forward pass for trajectory A
        outputs_A = model(
            input_ids=inputs["input_ids_A"],
            attention_mask=inputs["attention_mask_A"],
            pixel_values=inputs.get("pixel_values_A"),
            pixel_values_videos=inputs.get("pixel_values_videos_A"),
            image_grid_thw=inputs.get("image_grid_thw_A"),
            video_grid_thw=inputs.get("video_grid_thw_A"),
            prediction_type="preference"
        )
        
        # Forward pass for trajectory B
        outputs_B = model(
            input_ids=inputs["input_ids_B"],
            attention_mask=inputs["attention_mask_B"],
            pixel_values=inputs.get("pixel_values_B"),
            pixel_values_videos=inputs.get("pixel_values_videos_B"),
            image_grid_thw=inputs.get("image_grid_thw_B"),
            video_grid_thw=inputs.get("video_grid_thw_B"),
            prediction_type="preference"
        )
        
        # Get preference scores
        score_A = outputs_A.logits.squeeze(-1)
        score_B = outputs_B.logits.squeeze(-1)
        
        # Get preference labels (1 if A is preferred, 0 if B is preferred)
        preference_labels = inputs["preference_labels"]
        
        # Bradley-Terry model: P(A > B) = sigmoid(score_A - score_B)
        preference_logits = score_A - score_B
        loss = F.binary_cross_entropy_with_logits(preference_logits, preference_labels.float())
        
        if return_outputs:
            return loss, {"score_A": score_A, "score_B": score_B, "preference_labels": preference_labels}
        return loss
    
    def _compute_similarity_loss(self, model, inputs, return_outputs=False):
        """Compute similarity scoring loss (DPO-style)."""
        # Prepare model kwargs for chosen sequence (A vs B)
        chosen_kwargs = {
            "input_ids": inputs["input_ids_chosen"],
            "attention_mask": inputs["attention_mask_chosen"],
            "prediction_type": "similarity"
        }
        
        # Add vision inputs for chosen if they exist
        if "pixel_values_chosen" in inputs:
            chosen_kwargs["pixel_values"] = inputs["pixel_values_chosen"]
        if "pixel_values_videos_chosen" in inputs:
            chosen_kwargs["pixel_values_videos"] = inputs["pixel_values_videos_chosen"]
        if "image_grid_thw_chosen" in inputs:
            chosen_kwargs["image_grid_thw"] = inputs["image_grid_thw_chosen"]
        if "video_grid_thw_chosen" in inputs:
            chosen_kwargs["video_grid_thw"] = inputs["video_grid_thw_chosen"]
        
        # Forward pass for chosen sequence (A vs B)
        outputs_chosen = model(**chosen_kwargs)
        
        # Prepare model kwargs for rejected sequence (A vs C)
        rejected_kwargs = {
            "input_ids": inputs["input_ids_rejected"],
            "attention_mask": inputs["attention_mask_rejected"],
            "prediction_type": "similarity"
        }
        
        # Add vision inputs for rejected if they exist
        if "pixel_values_rejected" in inputs:
            rejected_kwargs["pixel_values"] = inputs["pixel_values_rejected"]
        if "pixel_values_videos_rejected" in inputs:
            rejected_kwargs["pixel_values_videos"] = inputs["pixel_values_videos_rejected"]
        if "image_grid_thw_rejected" in inputs:
            rejected_kwargs["image_grid_thw"] = inputs["image_grid_thw_rejected"]
        if "video_grid_thw_rejected" in inputs:
            rejected_kwargs["video_grid_thw"] = inputs["video_grid_thw_rejected"]
        
        # Forward pass for rejected sequence (A vs C)
        outputs_rejected = model(**rejected_kwargs)
        
        # Extract similarity scores
        score_chosen = outputs_chosen.logits.squeeze(-1)
        score_rejected = outputs_rejected.logits.squeeze(-1)
        
        # Compute DPO loss
        loss = -F.logsigmoid(self.beta * (score_chosen - score_rejected)).mean()
        
        if return_outputs:
            return loss, {"score_chosen": score_chosen, "score_rejected": score_rejected}
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step for RFM format that handles all three prediction types.
        """
        model.eval()
        
        with torch.no_grad():
            # Determine prediction type from inputs
            prediction_type = inputs.get("prediction_type", "similarity")
            
            # Compute the loss using our custom compute_loss method
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.detach().mean()
            
            if prediction_type == "progress":
                # For progress prediction, return predicted vs target progress
                predicted_progress = outputs["predicted_progress"]
                target_progress = outputs["target_progress"]
                logits = torch.stack([predicted_progress, target_progress], dim=-1)
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
                
            elif prediction_type == "preference":
                # For preference prediction, return preference scores
                score_A = outputs["score_A"]
                score_B = outputs["score_B"]
                logits = torch.stack([score_A, score_B], dim=-1)
                labels = outputs["preference_labels"].long()
                
            else:  # similarity
                # For similarity prediction, return chosen vs rejected scores
                score_chosen = outputs["score_chosen"]
                score_rejected = outputs["score_rejected"]
                logits = torch.stack([score_chosen, score_rejected], dim=-1)
                # Create dummy labels for similarity (no ground truth labels)
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
            
            print(f"DEBUG: prediction_step ({prediction_type}) - loss: {loss.shape}, logits: {logits.shape}, labels: {labels.shape}")
            
        return (loss, logits, labels)


def compute_metrics(eval_prediction):
    """
    Compute metrics for RFM evaluation across all three prediction types.
    This function is passed to the Trainer.
    """
    print(f"DEBUG: compute_metrics called with eval_prediction: {type(eval_prediction)}")
    print(f"DEBUG: eval_prediction.predictions shape: {eval_prediction.predictions.shape if eval_prediction.predictions is not None else None}")
    print(f"DEBUG: eval_prediction.label_ids shape: {eval_prediction.label_ids.shape if eval_prediction.label_ids is not None else None}")
    
    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    
    if predictions is not None and len(predictions.shape) >= 2:
        # predictions should be [batch_size, 2] for all prediction types
        score_1 = predictions[:, 0]
        score_2 = predictions[:, 1]
        
        # Determine prediction type based on the context (this would need to be passed in metadata)
        # For now, we'll compute metrics for all types and let the user interpret them
        
        # Progress prediction metrics (score_1 = predicted, score_2 = target)
        progress_mse = ((score_1 - score_2) ** 2).mean()
        progress_mae = abs(score_1 - score_2).mean()
        
        # Preference prediction metrics (score_1 = A, score_2 = B)
        if label_ids is not None:
            # If we have preference labels, compute preference accuracy
            preference_logits = score_1 - score_2
            preference_probs = 1 / (1 + np.exp(-preference_logits))
            predicted_preferences = (preference_probs > 0.5).astype(float)
            preference_accuracy = (predicted_preferences == label_ids.astype(float)).mean()
        else:
            preference_accuracy = None
        
        # Similarity prediction metrics (score_1 = chosen, score_2 = rejected)
        similarity_accuracy = (score_1 > score_2).astype(float).mean()
        similarity_diff = score_1 - score_2
        
        metrics = {
            # Progress metrics
            "progress_mse": progress_mse,
            "progress_mae": progress_mae,
            
            # Preference metrics
            "preference_accuracy": preference_accuracy,
            "avg_score_A": score_1.mean(),
            "avg_score_B": score_2.mean(),
            
            # Similarity metrics
            "similarity_accuracy": similarity_accuracy,
            "similarity_diff": similarity_diff.mean(),
            "avg_score_chosen": score_1.mean(),
            "avg_score_rejected": score_2.mean(),
        }
        
        print(f"DEBUG: computed metrics: {metrics}")
        return metrics
    else:
        print(f"DEBUG: predictions is None or wrong shape: {predictions}")
        return {}





def setup_model_and_processor(cfg: ExperimentConfig):
    """Shared function to set up model, processor, and tokenizer for both training and evaluation"""
    
    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(
        cfg.model.base_model_id, trust_remote_code=cfg.model.trust_remote_code
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Create a fresh model instance
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.model.base_model_id)
    
    # Add RFM special tokens if they don't exist
    special_tokens = ["<|split_token|>", "<|reward_token|>", "<|progress_token|>", "<|pref_token|>"]
    for token in special_tokens:
        if token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
            print(f"Added special token: {token}")
    
    # Resize token embeddings if new tokens were added
    if len(processor.tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(processor.tokenizer))
        print(f"Resized token embeddings to {len(processor.tokenizer)}")
    
    # Initialize RFM model wrapper
    print(f"Initializing RFM model wrapper...")
    rfm_model = RFMModel(
        config=base_model.config, tokenizer=processor.tokenizer
    )
    print(f"Loading base model state dict...")
    rfm_model.model.load_state_dict(base_model.state_dict())
    
    return processor, rfm_model


def setup_peft_model(rfm_model, cfg: ExperimentConfig):
    """Shared function to apply PEFT configuration to the model"""
    
    if cfg.peft.use_peft:
        print("Using PEFT/LoRA training...")
        lora_config = LoraConfig(
            r=cfg.peft.r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=cfg.peft.target_modules,
            lora_dropout=cfg.peft.lora_dropout,
            bias=cfg.peft.bias,
        )
        peft_rfm_model = get_peft_model(rfm_model, lora_config)
        for name, param in peft_rfm_model.named_parameters():
            if any(head in name for head in ["progress_head", "preference_head", "similarity_head"]):
                param.requires_grad = True
        if cfg.logging.print_trainable_parameters:
            peft_rfm_model.print_trainable_parameters()
        return peft_rfm_model
    else:
        print("Using full model training (no PEFT)...")
        peft_rfm_model = rfm_model
        # Configure which parts of the model to train based on config
        for name, param in peft_rfm_model.named_parameters():
            # Always train the prediction heads
            if any(head in name for head in ["progress_head", "preference_head", "similarity_head"]):
                param.requires_grad = cfg.peft.train_value_head
            # Train vision encoder if specified
            elif "visual" in name or "vision" in name:
                param.requires_grad = cfg.peft.train_vision_encoder
            # Train language model if specified
            elif "model" in name and not ("visual" in name or "vision" in name):
                param.requires_grad = cfg.peft.train_language_model
            # Default: train if language model training is enabled
            else:
                param.requires_grad = cfg.peft.train_language_model
        
        if cfg.logging.print_trainable_parameters:
            # Count trainable parameters manually
            trainable_params = sum(p.numel() for p in peft_rfm_model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in peft_rfm_model.parameters())
            print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
            print(f"Training configuration:")
            print(f"  - Vision encoder: {cfg.peft.train_vision_encoder}")
            print(f"  - Language model: {cfg.peft.train_language_model}")
            print(f"  - Prediction heads: {cfg.peft.train_value_head}")
        
        return peft_rfm_model


def create_training_arguments(cfg: ExperimentConfig, output_dir: str, is_eval: bool = False):
    """Shared function to create TrainingArguments for both training and evaluation"""
    
    # Base arguments that are the same for both training and evaluation
    base_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "save_strategy": cfg.training.save_strategy,
        "logging_steps": cfg.training.logging_steps,
        "save_steps": cfg.training.save_steps,
        "bf16": cfg.training.bf16,
        "remove_unused_columns": cfg.training.remove_unused_columns,
        "gradient_checkpointing": cfg.training.gradient_checkpointing,
        "fsdp": cfg.training.fsdp,
        "fsdp_config": cfg.training.fsdp_config,
        "dataloader_pin_memory": cfg.training.dataloader_pin_memory,
        "dataloader_num_workers": cfg.training.dataloader_num_workers,
        "ddp_find_unused_parameters": cfg.training.ddp_find_unused_parameters,
        "ddp_bucket_cap_mb": cfg.training.ddp_bucket_cap_mb,
        "save_safetensors": True,
        "save_total_limit": 2,
    }
    
    if is_eval:
        # Evaluation-specific arguments
        base_args.update({
            "per_device_eval_batch_size": 2,
            "num_train_epochs": -1,
            "max_steps": 1,
            "report_to": "none",
        })
    else:
        # Training-specific arguments
        base_args.update({
            "num_train_epochs": cfg.training.num_train_epochs if cfg.training.num_train_epochs is not None else 1,
            "max_steps": cfg.training.max_steps if cfg.training.max_steps is not None else -1,
            "report_to": ["wandb"] if cfg.logging.use_wandb else [],
        })
    
    return TrainingArguments(**base_args)


def train(cfg: ExperimentConfig):
    # If cfg is a string (config path), load it
    if isinstance(cfg, str):
        cfg = ExperimentConfig.from_yaml(cfg)
    
    # Initialize wandb if enabled
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.logging.wandb_run_name,
            config={
                "model": {
                    "base_model_id": cfg.model.base_model_id,
                    "torch_dtype": cfg.model.torch_dtype,
                },
                "peft": {
                    "use_peft": cfg.peft.use_peft,
                    "r": cfg.peft.r,
                    "lora_alpha": cfg.peft.lora_alpha,
                    "train_vision_encoder": cfg.peft.train_vision_encoder,
                    "train_language_model": cfg.peft.train_language_model,
                    "train_value_head": cfg.peft.train_value_head,
                },
                "training": {
                    "learning_rate": cfg.training.learning_rate,
                    "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
                    "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
                    "num_train_epochs": cfg.training.num_train_epochs,
                    "max_steps": cfg.training.max_steps,
                    "beta": cfg.training.beta,
                },
                "prompt": {
                    "discriminator": cfg.prompt.discriminator,
                }
            }
        )
        print(f"Wandb initialized: {wandb.run.name}")
    
    # Set memory management
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Use the shared function to set up model and processor
    processor, rfm_model = setup_model_and_processor(cfg)

    # Apply PEFT if enabled
    peft_rfm_model = setup_peft_model(rfm_model, cfg)
    
    # Import DataGenerator for creating Sample/Batch objects
    from data.data_generator import DataGenerator
    
    # Create DataGenerator for training
    data_generator = DataGenerator(
        dataset_path=cfg.training.dpo_dataset_path,
        batch_size=cfg.training.per_device_train_batch_size,
        preference_ratio=1.0,  # Use only preference samples for DPO training
        comparative_ratio=0.0,
        progress_ratio=0.0,
        max_frames=cfg.training.video_max_frames,
        shuffle=True,
        seed=42
    )
    
    # Create training arguments from config
    training_args = create_training_arguments(cfg, cfg.training.output_dir)
    
    # Use the new BatchCollator for processing samples
    batch_collator = BatchCollator(
        processor=processor,
        max_length=cfg.training.max_seq_length
    )
    
    # Create the dataset
    train_dataset = DataGeneratorDataset(data_generator, num_batches=1000)
    
    trainer = RFMTrainer(
        model=peft_rfm_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=batch_collator,
        beta=cfg.training.beta,
        compute_metrics=compute_metrics,  # Pass the compute_metrics function
    )
    
    print(f"Training from checkpoint: {cfg.training.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.save_model(cfg.training.output_dir)
    print(f"Training complete! Check {cfg.training.output_dir} for checkpoints and final model.")


def evaluate(cfg: ExperimentConfig):
    """Evaluate the trained RFM model using a subset of training data"""
    print("--- Evaluating RFM Model ---")
    
    # Use the shared function to set up model and processor
    processor, rfm_model = setup_model_and_processor(cfg)
    
    # Apply PEFT configuration (same as training) to ensure parameter groups match
    model = setup_peft_model(rfm_model, cfg)
    
    # Import DataGenerator for creating Sample/Batch objects
    from data.data_generator import DataGenerator
    
    # Create DataGenerator for evaluation
    eval_data_generator = DataGenerator(
        dataset_path=cfg.evaluation.eval_dataset_path,
        batch_size=2,  # Small batch size for evaluation
        preference_ratio=1.0,  # Use only preference samples for DPO evaluation
        comparative_ratio=0.0,
        progress_ratio=0.0,
        max_frames=cfg.training.video_max_frames,
        shuffle=False,  # No shuffling for evaluation
        seed=42
    )
    
    print(f"Using DataGenerator for evaluation with {cfg.evaluation.eval_subset_size} examples")
    
    # Create evaluation dataset using DataGenerator
    eval_dataset = DataGeneratorDataset(eval_data_generator, num_batches=cfg.evaluation.eval_subset_size)
    
    # Use the shared function to create training arguments for evaluation
    eval_args = create_training_arguments(cfg, "./eval_output", is_eval=True)
    
    # Use the new BatchCollator for evaluation
    batch_collator = BatchCollator(
        processor=processor,
        max_length=cfg.training.max_seq_length
    )
    
    # Initialize the Trainer
    # The trainer handles all the distributed complexity and FSDP loading
    print(f"DEBUG: Creating trainer with compute_metrics: {compute_metrics}")
    trainer = RFMTrainer(
        model=model,  # Use the PEFT-configured model
        args=eval_args,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        data_collator=batch_collator,
        beta=cfg.training.beta,
        compute_metrics=compute_metrics,  # Pass the compute_metrics function
    )
    print(f"DEBUG: Trainer created, compute_metrics: {trainer.compute_metrics}")
    
    # Load the checkpoint using trainer's resume_from_checkpoint feature
    # This will automatically load all weights including the base model into RFMModel
    print(f"Loading checkpoint from: {cfg.evaluation.model_path}")
    
    # Now that training arguments match, we can use resume_from_checkpoint safely
    trainer.train(resume_from_checkpoint=cfg.evaluation.model_path)
    
    # Run evaluation using trainer's evaluation infrastructure
    print("Running evaluation...")
    
    # Use trainer's evaluation method which properly handles FSDP
    eval_results = trainer.evaluate()
    
    # Only the main process should print the results
    if trainer.is_world_process_zero():
        print(f"Evaluation results: {eval_results}")
        print("\n=== Evaluation Results ===")
        
        # Helper function to safely format metrics
        def safe_format(metric_name, default_value=0.0):
            value = eval_results.get(metric_name, default_value)
            if isinstance(value, (int, float)):
                return f"{value:.6f}"
            else:
                return str(value)
        
        print(f"Evaluation Loss: {safe_format('eval_loss')}")
        print(f"Accuracy (B > C): {safe_format('eval_accuracy')} ({safe_format('eval_accuracy', 0)}%)")
        print(f"Average Reward Difference (B - C): {safe_format('eval_reward_diff')}")
        print(f"Average Reward for B (chosen): {safe_format('eval_avg_reward_chosen')}")
        print(f"Average Reward for C (rejected): {safe_format('eval_avg_reward_rejected')}")
        
        # Print interpretation
        accuracy = eval_results.get('eval_accuracy', 0)
        if isinstance(accuracy, (int, float)) and accuracy > 0.5:
            print(f"✅ Model correctly prefers B over C in {accuracy*100:.1f}% of cases")
        else:
            print(f"❌ Model incorrectly prefers C over B in {(1-accuracy)*100:.1f}% of cases")
    
    return eval_results


@wrap()
def main(cfg: ExperimentConfig):
    print(f'RFM Experiment - Mode: {cfg.mode}')
    
    if cfg.mode == "train":
        print(f'Training RFM model on LIBERO dataset...')
        print(f'\tModel: {cfg.model.base_model_id}')
        print(f'\tOutput directory: {cfg.training.output_dir}')
        print(f'\tDataset: {cfg.training.dpo_dataset_path}')
        print(f'\tBase directory: {cfg.training.base_dir}')
        print(f'\tBatch size: {cfg.training.per_device_train_batch_size}')
        print(f'\tLearning rate: {cfg.training.learning_rate}')
        print(f'\tEpochs: {cfg.training.num_train_epochs}')
        print(f'\tVideo max frames: {cfg.training.video_max_frames}')
        print(f'\tVideo frame sampling: {cfg.training.video_frame_sampling}')
        print(f'\tPrompt: {cfg.prompt.discriminator}')
        
        train(cfg)
        
    elif cfg.mode == "evaluate":
        print(f'Evaluating RFM model...')
        print(f'\tModel path: {cfg.evaluation.model_path}')
        print(f'\tEvaluation subset size: {cfg.evaluation.eval_subset_size}')
        
        evaluate(cfg)
        
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be 'train' or 'evaluate'")


if __name__ == "__main__":
    main()
