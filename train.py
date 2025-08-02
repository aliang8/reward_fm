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
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from PIL import Image
import json
import os
import yaml
from typing import List, Dict, Optional, Union, Any
from peft import get_peft_model, LoraConfig, PeftModel
from dpo_collator import DPOCollator
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


class BatchDPOCollator:
    """Batch DPO collator that processes multiple examples efficiently using batch processing"""
    
    def __init__(self, processor, pad_token_id: int, max_length: int = 1024):
        self.processor = processor
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate a batch of DPO examples using efficient batch processing.
        
        Args:
            batch: List of dictionaries, each containing:
                - conversation_chosen_1: First conversation for chosen pair (prompt + reference video)
                - conversation_chosen_2: Second conversation for chosen pair (prompt + candidate video + reward token)
                - conversation_rejected_1: First conversation for rejected pair (prompt + reference video)
                - conversation_rejected_2: Second conversation for rejected pair (prompt + candidate video + reward token)
        
        Returns:
            Dictionary with batched inputs for both chosen and rejected sequences
        """
        # Process each example individually first, then batch the results
        chosen_inputs_list = []
        rejected_inputs_list = []
        
        for item in batch:
            # Process chosen conversations (A vs B)
            # First conversation: prompt + reference video A
            conversation_chosen_1 = item["conversation_chosen_1"]
            
            # Extract video paths from the conversations
            video_paths = None
            video_paths_2 = None
            
            # Extract video paths from chosen conversations
            for content in conversation_chosen_1[0]["content"]:
                if content.get("type") == "video":
                    video_paths = content["video"]
                    break
            
            conversation_chosen_2 = item["conversation_chosen_2"]
            for content in conversation_chosen_2[0]["content"]:
                if content.get("type") == "video":
                    video_paths_2 = content["video"]
                    break
            
            if video_paths is None or video_paths_2 is None:
                raise ValueError("Missing video paths in chosen conversations")
            
            # Create a single conversation with both videos for chosen
            combined_conversation_chosen = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are shown three video sequences of robot trajectories. Sequences A and B are from the same task, while sequence C is from a different task. Which sequence (A or B) shows a better trajectory for the task?"},
                        {"type": "video", "video": video_paths},  # Reference video A
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "video", "video": video_paths_2},  # Candidate video B
                        {"type": "text", "text": "<|reward_token|>"}
                    ]
                }
            ]
            
            # Debug: Print the conversation structure
            # print(f"DEBUG: Combined conversation has {len(combined_conversation_chosen[0]['content'])} content items")
            # for i, content in enumerate(combined_conversation_chosen[0]['content']):
            #     print(f"  Content {i}: {content['type']}")
            
            # Process the combined conversation for both text and video
            combined_text_chosen = self.processor.apply_chat_template(
                combined_conversation_chosen, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            
            combined_image_inputs_chosen, combined_video_inputs_chosen, combined_video_kwargs_chosen = process_vision_info(
                combined_conversation_chosen, return_video_kwargs=True
            )
            
            # # Debug: Print video processing results
            # print(f"DEBUG: Video inputs type: {type(combined_video_inputs_chosen)}")
            # if combined_video_inputs_chosen is not None:
            #     if isinstance(combined_video_inputs_chosen, list):
            #         print(f"DEBUG: Video inputs is a list with {len(combined_video_inputs_chosen)} items")
            #         for i, video_input in enumerate(combined_video_inputs_chosen):
            #             print(f"  Video input {i} shape: {video_input.shape if hasattr(video_input, 'shape') else type(video_input)}")
            #     else:
            #         print(f"DEBUG: Video inputs shape: {combined_video_inputs_chosen.shape}")
            # else:
            #     print("DEBUG: Video inputs is None")
            # print(f"DEBUG: Video kwargs keys: {list(combined_video_kwargs_chosen.keys()) if combined_video_kwargs_chosen else 'None'}")
            
            inputs_chosen = self.processor(
                text=[combined_text_chosen],
                images=combined_image_inputs_chosen,
                videos=combined_video_inputs_chosen,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **combined_video_kwargs_chosen,
            )
            
            # Debug: Check the processor output
            # print(f"DEBUG: Processor output keys: {list(inputs_chosen.keys())}")
            # if "video_grid_thw" in inputs_chosen:
            #     print(f"DEBUG: video_grid_thw shape: {inputs_chosen['video_grid_thw'].shape}")
            #     print(f"DEBUG: video_grid_thw content: {inputs_chosen['video_grid_thw']}")
            # if "pixel_values_videos" in inputs_chosen:
            #     video_values = inputs_chosen['pixel_values_videos']
            #     if isinstance(video_values, list):
            #         print(f"DEBUG: pixel_values_videos is a list with {len(video_values)} items")
            #         for i, video_input in enumerate(video_values):
            #             print(f"  Video input {i} shape: {video_input.shape if hasattr(video_input, 'shape') else type(video_input)}")
            #     else:
            #         print(f"DEBUG: pixel_values_videos shape: {video_values.shape}")
            # else:
            #     print("DEBUG: pixel_values_videos not found")
            
            # Debug: Count video tokens in text
            # video_token_count = combined_text_chosen.count("<|video_pad|>")
            # print(f"DEBUG: Video token count in text: {video_token_count}")
            
            # Process rejected conversations (A vs C)
            # First conversation: prompt + reference video A
            conversation_rejected_1 = item["conversation_rejected_1"]
            
            # Extract video paths from rejected conversations
            video_paths_rejected_1 = None
            video_paths_rejected_2 = None
            
            for content in conversation_rejected_1[0]["content"]:
                if content.get("type") == "video":
                    video_paths_rejected_1 = content["video"]
                    break
            
            conversation_rejected_2 = item["conversation_rejected_2"]
            for content in conversation_rejected_2[0]["content"]:
                if content.get("type") == "video":
                    video_paths_rejected_2 = content["video"]
                    break
            
            if video_paths_rejected_1 is None or video_paths_rejected_2 is None:
                raise ValueError("Missing video paths in rejected conversations")
            
            # Create a single conversation with both videos for rejected
            combined_conversation_rejected = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are shown three video sequences of robot trajectories. Sequences A and B are from the same task, while sequence C is from a different task. Which sequence (A or B) shows a better trajectory for the task?"},
                        {"type": "video", "video": video_paths_rejected_1},  # Reference video A
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "video", "video": video_paths_rejected_2},  # Candidate video C
                        {"type": "text", "text": "<|reward_token|>"}
                    ]
                }
            ]
            
            # Process the combined conversation for both text and video
            combined_text_rejected = self.processor.apply_chat_template(
                combined_conversation_rejected, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            
            combined_image_inputs_rejected, combined_video_inputs_rejected, combined_video_kwargs_rejected = process_vision_info(
                combined_conversation_rejected, return_video_kwargs=True
            )
            
            inputs_rejected = self.processor(
                text=[combined_text_rejected],
                images=combined_image_inputs_rejected,
                videos=combined_video_inputs_rejected,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **combined_video_kwargs_rejected,
            )
            
            # Debug: Check the processor output for rejected
            # print(f"DEBUG: Rejected processor output keys: {list(inputs_rejected.keys())}")
            # if "video_grid_thw" in inputs_rejected:
            #     print(f"DEBUG: Rejected video_grid_thw shape: {inputs_rejected['video_grid_thw'].shape}")
            #     print(f"DEBUG: Rejected video_grid_thw content: {inputs_rejected['video_grid_thw']}")
            # if "pixel_values_videos" in inputs_rejected:
            #     video_values = inputs_rejected['pixel_values_videos']
            #     if isinstance(video_values, list):
            #         print(f"DEBUG: Rejected pixel_values_videos is a list with {len(video_values)} items")
            #         for i, video_input in enumerate(video_values):
            #             print(f"  Video input {i} shape: {video_input.shape if hasattr(video_input, 'shape') else type(video_input)}")
            #     else:
            #         print(f"DEBUG: Rejected pixel_values_videos shape: {video_values.shape}")
            # else:
            #     print("DEBUG: Rejected pixel_values_videos not found")
            
            # # Debug: Count video tokens in rejected text
            # video_token_count_rejected = combined_text_rejected.count("<|video_pad|>")
            # print(f"DEBUG: Rejected video token count in text: {video_token_count_rejected}")
            
            # Store the processed inputs
            chosen_inputs_list.append(inputs_chosen)
            rejected_inputs_list.append(inputs_rejected)
        
        # Now batch the processed inputs
        # Initialize with the first example
        batched_chosen = {
            "input_ids": chosen_inputs_list[0]["input_ids"],
            "attention_mask": chosen_inputs_list[0]["attention_mask"],
        }
        batched_rejected = {
            "input_ids": rejected_inputs_list[0]["input_ids"],
            "attention_mask": rejected_inputs_list[0]["attention_mask"],
        }
        
        # Add vision inputs for chosen if they exist
        if "pixel_values" in chosen_inputs_list[0]:
            batched_chosen["pixel_values"] = chosen_inputs_list[0]["pixel_values"]
        if "pixel_values_videos" in chosen_inputs_list[0]:
            batched_chosen["pixel_values_videos"] = chosen_inputs_list[0]["pixel_values_videos"]
        if "image_grid_thw" in chosen_inputs_list[0]:
            batched_chosen["image_grid_thw"] = chosen_inputs_list[0]["image_grid_thw"]
        if "video_grid_thw" in chosen_inputs_list[0]:
            video_grid_thw = chosen_inputs_list[0]["video_grid_thw"]
            # Ensure proper shape for video_grid_thw
            if video_grid_thw.dim() == 3:  # (batch, time, height, width)
                video_grid_thw = video_grid_thw.unsqueeze(1)  # Add time dimension if needed
            batched_chosen["video_grid_thw"] = video_grid_thw
        
        # Add vision inputs for rejected if they exist
        if "pixel_values" in rejected_inputs_list[0]:
            batched_rejected["pixel_values"] = rejected_inputs_list[0]["pixel_values"]
        if "pixel_values_videos" in rejected_inputs_list[0]:
            batched_rejected["pixel_values_videos"] = rejected_inputs_list[0]["pixel_values_videos"]
        if "image_grid_thw" in rejected_inputs_list[0]:
            batched_rejected["image_grid_thw"] = rejected_inputs_list[0]["image_grid_thw"]
        if "video_grid_thw" in rejected_inputs_list[0]:
            video_grid_thw = rejected_inputs_list[0]["video_grid_thw"]
            # Ensure proper shape for video_grid_thw
            if video_grid_thw.dim() == 3:  # (batch, time, height, width)
                video_grid_thw = video_grid_thw.unsqueeze(1)  # Add time dimension if needed
            batched_rejected["video_grid_thw"] = video_grid_thw
        
        # Concatenate remaining examples
        for i in range(1, len(chosen_inputs_list)):
            # Concatenate chosen inputs
            batched_chosen["input_ids"] = torch.cat([batched_chosen["input_ids"], chosen_inputs_list[i]["input_ids"]], dim=0)
            batched_chosen["attention_mask"] = torch.cat([batched_chosen["attention_mask"], chosen_inputs_list[i]["attention_mask"]], dim=0)
            
            # Concatenate vision inputs for chosen if they exist
            if "pixel_values" in chosen_inputs_list[i]:
                batched_chosen["pixel_values"] = torch.cat([batched_chosen["pixel_values"], chosen_inputs_list[i]["pixel_values"]], dim=0)
            if "pixel_values_videos" in chosen_inputs_list[i]:
                batched_chosen["pixel_values_videos"] = torch.cat([batched_chosen["pixel_values_videos"], chosen_inputs_list[i]["pixel_values_videos"]], dim=0)
            if "image_grid_thw" in chosen_inputs_list[i]:
                batched_chosen["image_grid_thw"] = torch.cat([batched_chosen["image_grid_thw"], chosen_inputs_list[i]["image_grid_thw"]], dim=0)
            if "video_grid_thw" in chosen_inputs_list[i]:
                video_grid_thw = chosen_inputs_list[i]["video_grid_thw"]
                if video_grid_thw.dim() == 3:
                    video_grid_thw = video_grid_thw.unsqueeze(1)
                batched_chosen["video_grid_thw"] = torch.cat([batched_chosen["video_grid_thw"], video_grid_thw], dim=0)
            
            # Concatenate rejected inputs
            batched_rejected["input_ids"] = torch.cat([batched_rejected["input_ids"], rejected_inputs_list[i]["input_ids"]], dim=0)
            batched_rejected["attention_mask"] = torch.cat([batched_rejected["attention_mask"], rejected_inputs_list[i]["attention_mask"]], dim=0)
            
            # Concatenate vision inputs for rejected if they exist
            if "pixel_values" in rejected_inputs_list[i]:
                batched_rejected["pixel_values"] = torch.cat([batched_rejected["pixel_values"], rejected_inputs_list[i]["pixel_values"]], dim=0)
            if "pixel_values_videos" in rejected_inputs_list[i]:
                batched_rejected["pixel_values_videos"] = torch.cat([batched_rejected["pixel_values_videos"], rejected_inputs_list[i]["pixel_values_videos"]], dim=0)
            if "image_grid_thw" in rejected_inputs_list[i]:
                batched_rejected["image_grid_thw"] = torch.cat([batched_rejected["image_grid_thw"], rejected_inputs_list[i]["image_grid_thw"]], dim=0)
            if "video_grid_thw" in rejected_inputs_list[i]:
                video_grid_thw = rejected_inputs_list[i]["video_grid_thw"]
                if video_grid_thw.dim() == 3:
                    video_grid_thw = video_grid_thw.unsqueeze(1)
                batched_rejected["video_grid_thw"] = torch.cat([batched_rejected["video_grid_thw"], video_grid_thw], dim=0)
        
        # Build final result dictionary
        result = {
            "input_ids_chosen": batched_chosen["input_ids"],
            "attention_mask_chosen": batched_chosen["attention_mask"],
            "input_ids_rejected": batched_rejected["input_ids"],
            "attention_mask_rejected": batched_rejected["attention_mask"],
        }
        
        # Add vision inputs for chosen if they exist
        if "pixel_values" in batched_chosen:
            result["pixel_values_chosen"] = batched_chosen["pixel_values"]
        if "pixel_values_videos" in batched_chosen:
            result["pixel_values_videos_chosen"] = batched_chosen["pixel_values_videos"]
        if "image_grid_thw" in batched_chosen:
            result["image_grid_thw_chosen"] = batched_chosen["image_grid_thw"]
        if "video_grid_thw" in batched_chosen:
            result["video_grid_thw_chosen"] = batched_chosen["video_grid_thw"]
        
        # Add vision inputs for rejected if they exist
        if "pixel_values" in batched_rejected:
            result["pixel_values_rejected"] = batched_rejected["pixel_values"]
        if "pixel_values_videos" in batched_rejected:
            result["pixel_values_videos_rejected"] = batched_rejected["pixel_values_videos"]
        if "image_grid_thw" in batched_rejected:
            result["image_grid_thw_rejected"] = batched_rejected["image_grid_thw"]
        if "video_grid_thw" in batched_rejected:
            result["video_grid_thw_rejected"] = batched_rejected["video_grid_thw"]
        
        return result


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
    dpo_dataset_path: str = field(default="libero_dpo_dataset/libero_dpo_dataset.json")
    base_dir: str = field(default="libero_dpo_dataset")
    output_dir: str = field(default="./libero_reward_model_output")
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


# FSDPConfig removed - now part of TrainingConfig


@dataclass
class LoggingConfig:
    """Config for logging settings"""
    print_trainable_parameters: bool = field(default=True)
    save_model: bool = field(default=True)
    save_processor: bool = field(default=True)
    # Wandb configuration
    use_wandb: bool = field(default=True, metadata={"help": "Whether to use Weights & Biases logging"})
    wandb_project: str = field(default="vlm-reward-model", metadata={"help": "Wandb project name"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Wandb entity/username"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "Wandb run name"})


@dataclass
class PromptConfig:
    """Config for prompt settings"""
    discriminator: str = field(default="You are shown three video sequences of robot trajectories. Sequences A and B are from the same task, while sequence C is from a different task. Which sequence (A or B) shows a better trajectory for the task?")


@dataclass
class EvaluationConfig:
    """Config for evaluation settings"""
    model_path: str = field(default="./libero_reward_model_output")
    eval_subset_size: int = field(default=10, metadata={"help": "Number of examples to use for evaluation"})
    eval_dataset_path: str = field(default="libero_dpo_dataset/libero_dpo_dataset.json")
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
    

class VLMRewardModel(PreTrainedModel):
    config_class = Qwen2_5_VLForConditionalGeneration.config_class

    def __init__(self, config, tokenizer):
        super().__init__(config)
        # The VLMRewardModel now owns and creates its submodules.
        # This is the standard pattern for PreTrainedModel.
        self.model = Qwen2_5_VLForConditionalGeneration(config)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

        # Ensure the value_head and base model have the same dtype
        self.model_dtype = self.model.dtype
        self.value_head = self.value_head.to(dtype=self.model_dtype)
        
        # The tokenizer can be attached after initialization if needed.
        self.tokenizer = tokenizer

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegates gradient checkpointing enabling to the base model."""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        """Delegates gradient checkpointing disabling to the base model."""
        self.model.gradient_checkpointing_disable(**kwargs)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Handle case where input_ids and attention_mask might be in kwargs
        if input_ids is None and "input_ids" in kwargs:
            input_ids = kwargs.pop("input_ids")
        if attention_mask is None and "attention_mask" in kwargs:
            attention_mask = kwargs.pop("attention_mask")
        
        # Handle nested argument structures (common in evaluation)
        if input_ids is None and "input_ids_chosen" in kwargs:
            input_ids = kwargs.pop("input_ids_chosen")
        if attention_mask is None and "attention_mask_chosen" in kwargs:
            attention_mask = kwargs.pop("attention_mask_chosen")
        
        # Handle case where arguments might be in a nested dict
        if input_ids is None and "inputs" in kwargs:
            inputs = kwargs.pop("inputs")
            if isinstance(inputs, dict):
                input_ids = inputs.get("input_ids")
                attention_mask = inputs.get("attention_mask")
        
        # Ensure required arguments are provided
        if input_ids is None:
            raise ValueError("input_ids is required")
        if attention_mask is None:
            raise ValueError("attention_mask is required")
        
        # Prepare model kwargs with all available inputs
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            **kwargs,
        }
        
        # Add vision inputs based on what's available
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values
        if pixel_values_videos is not None:
            model_kwargs["pixel_values_videos"] = pixel_values_videos
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw
        if video_grid_thw is not None:
            model_kwargs["video_grid_thw"] = video_grid_thw
        
        # Forward pass through the model
        outputs = self.model(**model_kwargs)
        last_hidden_state = outputs.hidden_states[-1]

        reward_token_id = self.tokenizer.convert_tokens_to_ids("<|reward_token|>")
        
        # Find all positions where <|reward_token|> appears
        reward_token_positions = []
        for i, seq_ids in enumerate(input_ids):
            # Find the last occurrence of reward_token_id in this sequence
            positions = (seq_ids == reward_token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                reward_token_positions.append(positions[-1].item())
            else:
                # Fallback to last token if reward token not found
                reward_token_positions.append(attention_mask[i].sum().item() - 1)
        reward_token_positions = torch.tensor(reward_token_positions, device=input_ids.device)
        
        # Extract hidden states at the <|reward_token|> positions
        reward_hidden_states = torch.gather(
            last_hidden_state,
            1,
            reward_token_positions.view(-1, 1, 1).expand(
                -1, -1, last_hidden_state.size(-1)
            ),
        ).squeeze(1)
        
        # For POLAR, we use a single linear head on the reward token hidden state
        # The value head expects input of size hidden_size (not 2*hidden_size)
        rewards = self.value_head(reward_hidden_states)
        return SequenceClassifierOutputWithPast(logits=rewards)


class VLMRewardTrainer(Trainer):
    def __init__(self, *args, beta=0.1, compute_metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.compute_metrics = compute_metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Prepare model kwargs for chosen sequence (A vs B)
        chosen_kwargs = {
            "input_ids": inputs["input_ids_chosen"],
            "attention_mask": inputs["attention_mask_chosen"],
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
        
        # Extract rewards from the <|reward_token|> positions
        reward_chosen = outputs_chosen.logits.squeeze(-1)
        reward_rejected = outputs_rejected.logits.squeeze(-1)
        
        # Compute DPO loss
        loss = -F.logsigmoid(self.beta * (reward_chosen - reward_rejected)).mean()
        if return_outputs:
            return (
                loss,
                {"reward_chosen": reward_chosen, "reward_rejected": reward_rejected},
            )
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step for DPO format that computes additional metrics.
        """
        model.eval()
        
        with torch.no_grad():
            # Compute the loss using our custom compute_loss method
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.detach().mean()
            
            # Extract rewards for accuracy computation
            # Prepare model kwargs for chosen sequence (A vs B)
            chosen_kwargs = {
                "input_ids": inputs["input_ids_chosen"],
                "attention_mask": inputs["attention_mask_chosen"],
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
            
            # Extract rewards
            reward_chosen = outputs_chosen.logits.squeeze(-1)
            reward_rejected = outputs_rejected.logits.squeeze(-1)
            
            # Return loss, logits (rewards), and labels (None for DPO)
            # For DPO, we return the rewards as logits
            logits = torch.stack([reward_chosen, reward_rejected], dim=-1)
            
            # Create dummy labels to ensure compute_metrics gets called
            # The Trainer expects label_ids to be not None to call compute_metrics
            batch_size = logits.shape[0]
            dummy_labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
            
            print(f"DEBUG: prediction_step returning - loss: {loss.shape}, logits: {logits.shape}, labels: {dummy_labels.shape}")
            
        return (loss, logits, dummy_labels)


def compute_metrics(eval_prediction):
    """
    Compute metrics for DPO evaluation.
    This function is passed to the Trainer.
    """
    print(f"DEBUG: compute_metrics called with eval_prediction: {type(eval_prediction)}")
    print(f"DEBUG: eval_prediction.predictions shape: {eval_prediction.predictions.shape if eval_prediction.predictions is not None else None}")
    print(f"DEBUG: eval_prediction.label_ids shape: {eval_prediction.label_ids.shape if eval_prediction.label_ids is not None else None}")
    
    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    
    # For DPO, we don't have traditional labels, but we can compute metrics from predictions
    if predictions is not None and len(predictions.shape) >= 2:
        # predictions should be [batch_size, 2] where [:, 0] is reward_chosen and [:, 1] is reward_rejected
        reward_chosen = predictions[:, 0]
        reward_rejected = predictions[:, 1]
        
        # Compute accuracy: B should be preferred over C (reward_chosen > reward_rejected)
        correct_predictions = (reward_chosen > reward_rejected).astype(float)
        accuracy = correct_predictions.mean()
        
        # Compute additional metrics
        reward_diff = reward_chosen - reward_rejected
        avg_reward_chosen = reward_chosen.mean()
        avg_reward_rejected = reward_rejected.mean()
        
        metrics = {
            "accuracy": accuracy,
            "reward_diff": reward_diff.mean(),
            "avg_reward_chosen": avg_reward_chosen,
            "avg_reward_rejected": avg_reward_rejected,
        }
        
        print(f"DEBUG: computed metrics: {metrics}")
        return metrics
    else:
        print(f"DEBUG: predictions is None or wrong shape: {predictions}")
        return {}


def load_dpo_dataset(dataset_path: str) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image(image_path: str, base_dir: str) -> Image.Image:
    """
    Load a single image (for backward compatibility).
    For trajectories, use load_trajectory_media instead.
    """
    full_path = os.path.join(base_dir, image_path)
    if os.path.exists(full_path):
        return Image.open(full_path).convert("RGB")
    return Image.new("RGB", (224, 224), "gray")


def preprocess_dpo_example(
    example: Dict,
    base_dir: str,
    prompt: str = None,
    max_frames: int = 8,
) -> Dict:
    # LIBERO dataset uses video sequences, so we expect video_*_path fields
    if "video_A_path" not in example or "video_B_path" not in example or "video_C_path" not in example:
        raise ValueError("LIBERO dataset requires video_A_path, video_B_path, and video_C_path fields")
    
    # Get file paths for video sequences and validate they exist
    trajectory_A_paths = [os.path.join(base_dir, path) for path in example["video_A_path"][:max_frames]]
    trajectory_B_paths = [os.path.join(base_dir, path) for path in example["video_B_path"][:max_frames]]
    trajectory_C_paths = [os.path.join(base_dir, path) for path in example["video_C_path"][:max_frames]]
    
    # Validate that all paths exist
    for path_list, name in [(trajectory_A_paths, "A"), (trajectory_B_paths, "B"), (trajectory_C_paths, "C")]:
        for path in path_list:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path not found for trajectory {name}: {path}")
        
    # Create POLAR-style conversations: prompt + reference video A + <|split_token|> + candidate video B + <|reward_token|>
    # We need to create separate conversations for each video since process_vision_info expects one video per conversation
    
    # For chosen pair (A vs B): A is reference, B is candidate
    # First conversation: prompt + reference video A
    conversation_chosen_1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video", 
                    "video": trajectory_A_paths,  # Reference trajectory
                    "max_pixels": 360 * 420,  # Add video processing parameters
                    "fps": 1.0,
                },
            ],
        }
    ]
    
    # Second conversation: candidate video B + reward token (no prompt)
    conversation_chosen_2 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": trajectory_B_paths,  # Candidate trajectory
                    "max_pixels": 360 * 420,  # Add video processing parameters
                    "fps": 1.0,
                },
                {"type": "text", "text": "<|reward_token|>"}
            ],
        }
    ]
    
    # For rejected pair (A vs C): A is reference, C is candidate
    # First conversation: prompt + reference video A
    conversation_rejected_1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video", 
                    "video": trajectory_A_paths,  # Reference trajectory
                    "max_pixels": 360 * 420,  # Add video processing parameters
                    "fps": 1.0,
                },
            ],
        }
    ]
    
    # Second conversation: candidate video C + reward token (no prompt)
    conversation_rejected_2 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": trajectory_C_paths,  # Candidate trajectory
                    "max_pixels": 360 * 420,  # Add video processing parameters
                    "fps": 1.0,
                },
                {"type": "text", "text": "<|reward_token|>"}
            ],
        }
    ]
    
    # Return the conversations directly for batch processing
    return {
        "conversation_chosen_1": conversation_chosen_1,
        "conversation_chosen_2": conversation_chosen_2,
        "conversation_rejected_1": conversation_rejected_1,
        "conversation_rejected_2": conversation_rejected_2,
    }


def create_dpo_dataset(
    dpo_data: List[Dict],
    base_dir: str,
    max_length: int,
    prompt_template: str = None,
    max_frames: int = 8,
) -> Dataset:
    processed_examples = []
    for ex in tqdm(dpo_data, desc="Processing DPO examples"):
        processed_examples.append(
            preprocess_dpo_example(ex, base_dir, prompt_template, max_frames)
        )
    return Dataset.from_list(processed_examples)


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
    
    # Add POLAR special tokens if they don't exist
    special_tokens = ["<|split_token|>", "<|reward_token|>"]
    for token in special_tokens:
        if token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
            print(f"Added special token: {token}")
    
    # Resize token embeddings if new tokens were added
    if len(processor.tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(processor.tokenizer))
        print(f"Resized token embeddings to {len(processor.tokenizer)}")
    
    # Initialize reward model wrapper
    print(f"Initializing reward model wrapper...")
    vlm_rm = VLMRewardModel(
        config=base_model.config, tokenizer=processor.tokenizer
    )
    print(f"Loading base model state dict...")
    vlm_rm.model.load_state_dict(base_model.state_dict())
    
    return processor, vlm_rm


def setup_peft_model(vlm_rm, cfg: ExperimentConfig):
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
        peft_reward_model = get_peft_model(vlm_rm, lora_config)
        for name, param in peft_reward_model.named_parameters():
            if "value_head" in name:
                param.requires_grad = True
        if cfg.logging.print_trainable_parameters:
            peft_reward_model.print_trainable_parameters()
        return peft_reward_model
    else:
        print("Using full model training (no PEFT)...")
        peft_reward_model = vlm_rm
        # Configure which parts of the model to train based on config
        for name, param in peft_reward_model.named_parameters():
            # Always train the value head
            if "value_head" in name:
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
            trainable_params = sum(p.numel() for p in peft_reward_model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in peft_reward_model.parameters())
            print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
            print(f"Training configuration:")
            print(f"  - Vision encoder: {cfg.peft.train_vision_encoder}")
            print(f"  - Language model: {cfg.peft.train_language_model}")
            print(f"  - Value head: {cfg.peft.train_value_head}")
        
        return peft_reward_model


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
    processor, vlm_rm = setup_model_and_processor(cfg)

    # Apply PEFT if enabled
    peft_reward_model = setup_peft_model(vlm_rm, cfg)
    
    dpo_data = load_dpo_dataset(cfg.training.dpo_dataset_path)
    train_dataset = create_dpo_dataset(
        dpo_data,
        cfg.training.base_dir,
        cfg.training.max_seq_length,
        cfg.prompt.discriminator,
        cfg.training.video_max_frames,
    )
    
    # Create training arguments from config
    training_args = create_training_arguments(cfg, cfg.training.output_dir)
    
    # Use the new BatchDPOCollator for efficient batch processing
    batch_collator = BatchDPOCollator(
        processor=processor,
        pad_token_id=processor.tokenizer.pad_token_id,
        max_length=cfg.training.max_seq_length
    )
    
    trainer = VLMRewardTrainer(
        model=peft_reward_model,
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
    """Evaluate the trained reward model using a subset of training data"""
    print("--- Evaluating VLM Reward Model ---")
    
    # Use the shared function to set up model and processor
    processor, vlm_rm = setup_model_and_processor(cfg)
    
    # Apply PEFT configuration (same as training) to ensure parameter groups match
    model = setup_peft_model(vlm_rm, cfg)
    
    # Load the evaluation dataset (separate from training)
    eval_dataset_path = cfg.evaluation.eval_dataset_path
    eval_base_dir = cfg.evaluation.eval_base_dir
    
    print(f"Loading evaluation dataset from: {eval_dataset_path}")
    dpo_data = load_dpo_dataset(eval_dataset_path)
    
    # Use a small subset for evaluation
    if cfg.evaluation.eval_subset_size == -1:
        eval_subset = dpo_data
    else:
        eval_subset = dpo_data[:cfg.evaluation.eval_subset_size]
    print(f"Using {len(eval_subset)} examples for evaluation")
    
    # Create evaluation dataset using the same pipeline
    eval_dataset = create_dpo_dataset(
        eval_subset,
        eval_base_dir,  # Use eval_base_dir instead of training base_dir
        cfg.training.max_seq_length,
        cfg.prompt.discriminator,
        cfg.training.video_max_frames,
    )
    
    # Use the shared function to create training arguments for evaluation
    eval_args = create_training_arguments(cfg, "./eval_output", is_eval=True)
    
    # Use the new BatchDPOCollator for evaluation
    batch_collator = BatchDPOCollator(
        processor=processor,
        pad_token_id=processor.tokenizer.pad_token_id,
        max_length=cfg.training.max_seq_length
    )
    
    # Initialize the Trainer
    # The trainer handles all the distributed complexity and FSDP loading
    print(f"DEBUG: Creating trainer with compute_metrics: {compute_metrics}")
    trainer = VLMRewardTrainer(
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
    # This will automatically load all weights including the base model into VLMRewardModel
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
    print(f'VLM Reward Model Experiment - Mode: {cfg.mode}')
    
    if cfg.mode == "train":
        print(f'Training VLM reward model on LIBERO dataset...')
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
        print(f'Evaluating VLM reward model...')
        print(f'\tModel path: {cfg.evaluation.model_path}')
        print(f'\tEvaluation subset size: {cfg.evaluation.eval_subset_size}')
        
        evaluate(cfg)
        
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be 'train' or 'evaluate'")


if __name__ == "__main__":
    main()
