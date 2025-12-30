# Molmo2 Integration Guide

This document summarizes the changes made to integrate Molmo2 (`allenai/Molmo2-4B`) into the RFM training pipeline, which was originally designed for Qwen VL models.

## Overview

Molmo2 has several architectural differences from Qwen that required custom handling:

1. **Embedding Layer Structure**: Uses `torch.nn.Parameter` instead of `nn.Embedding`
2. **Vision Token Structure**: Uses `<low_res_im_start>` + `<im_patch>` tokens instead of `<|vision_start|>`/`<|vision_end|>` pairs
3. **Input Parameters**: Uses `image_grids`, `image_token_pooling`, `image_num_crops` instead of `image_grid_thw`
4. **No Video Processor Attributes**: Doesn't have `temporal_patch_size` or `merge_size`

---

## Changes Made

### 1. Custom Token Embedding Resize (`rfm/utils/setup_utils.py`)

**Problem**: Molmo2's `Molmo2Embedding` stores embeddings as a raw `torch.nn.Parameter`, not an `nn.Embedding` module. The standard `resize_token_embeddings()` fails with:

```
AttributeError: 'Molmo2Embedding' object has no attribute 'weight'
```

**Solution**: Implemented custom resize logic that:
1. Extracts the embedding `Parameter` from `Molmo2Embedding.embedding`
2. Creates a new expanded `Parameter` tensor
3. Copies existing weights and initializes new tokens with mean embedding

```python
is_molmo = "Molmo" in cfg.base_model_id
if is_molmo:
    new_vocab_size = len(processor.tokenizer)
    _embed_layer = base_model.get_input_embeddings()
    
    if hasattr(_embed_layer, 'embedding'):
        old_embed_attr = _embed_layer.embedding
        
        # Case 1: embedding is a Parameter (raw tensor)
        if isinstance(old_embed_attr, torch.nn.Parameter):
            old_num_tokens, embedding_dim = old_embed_attr.shape
            
            # Create new parameter with expanded vocab
            new_embed_data = torch.zeros(new_vocab_size, embedding_dim, 
                                         device=old_embed_attr.device, 
                                         dtype=old_embed_attr.dtype)
            
            # Copy existing weights
            new_embed_data[:old_num_tokens] = old_embed_attr.data
            
            # Initialize new token embeddings using mean of existing embeddings
            mean_embedding = old_embed_attr.data.mean(dim=0)
            new_embed_data[old_num_tokens:] = mean_embedding.unsqueeze(0).expand(
                new_vocab_size - old_num_tokens, -1)
            
            # Replace the embedding Parameter
            _embed_layer.embedding = torch.nn.Parameter(new_embed_data)
            
            # Update config to reflect new vocab size
            base_model.config.text_config.vocab_size = new_vocab_size
```

---

### 2. Vision Token Handling (`rfm/models/rfm.py`)

**Problem**: Molmo2 uses a different token structure for images:

| Model | Image Start Token | Image End Token | Image Content |
|-------|------------------|-----------------|---------------|
| Qwen | `<\|vision_start\|>` | `<\|vision_end\|>` | Vision embeddings |
| Molmo2 | `<low_res_im_start>` | *(none)* | `<im_patch>` tokens |

The original code looked for `<|vision_end|>` which doesn't exist in Molmo2, causing:

```
AttributeError: 'bool' object has no attribute 'nonzero'
```

This occurred because `tokenizer.convert_tokens_to_ids("<low_res_im_end>")` returns `None`, and `tensor == None` returns a Python `bool` instead of a tensor.

**Solution**: For Molmo2, compute end positions by finding the last `<im_patch>` token for each image:

```python
# In _forward_qwen method
is_molmo = "Molmo" in self.base_model_id
if is_molmo:
    vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<low_res_im_start>")
    vision_end_token_id = None  # Molmo2 has no explicit end token
    im_patch_token_id = self.processor.tokenizer.convert_tokens_to_ids("<im_patch>")
else:
    vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_patch_token_id = None

# For Molmo2, compute vision_end_positions from im_patch tokens
if is_molmo and im_patch_token_id is not None:
    im_patch_positions = (seq_ids == im_patch_token_id).nonzero(as_tuple=True)[0]
    vision_end_positions = []
    for start_idx, start_pos in enumerate(vision_start_positions):
        patches_after_start = im_patch_positions[im_patch_positions > start_pos]
        if len(patches_after_start) > 0:
            # Find patches before next image start (or end of sequence)
            if start_idx + 1 < len(vision_start_positions):
                next_start = vision_start_positions[start_idx + 1].item()
                patches_for_this_image = patches_after_start[patches_after_start < next_start]
            else:
                patches_for_this_image = patches_after_start
            if len(patches_for_this_image) > 0:
                vision_end_positions.append(patches_for_this_image[-1])
    vision_end_positions = torch.tensor(vision_end_positions, device=seq_ids.device)
```

---

### 3. Hidden State Extraction (`rfm/models/rfm.py`)

**Problem**: The `_extract_hidden_states_from_token_pairs` method was hardcoded for Qwen's token structure.

**Solution**: Added Molmo2 mode that uses `<low_res_im_start>` and `<im_patch>` tokens:

```python
def _extract_hidden_states_from_token_pairs(self, hidden_state, input_ids):
    is_molmo = "Molmo" in self.base_model_id
    
    if "SmolVLM" in self.base_model_id:
        # SmolVLM mode: same token appears in pairs
        use_same_token = True
        use_molmo_mode = False
    elif is_molmo:
        # Molmo2 mode: <low_res_im_start> followed by <im_patch> tokens
        tokenizer = self.processor.tokenizer
        start_token = "<low_res_im_start>"
        patch_token = "<im_patch>"
        use_same_token = False
        use_molmo_mode = True
    else:
        # Qwen mode: different start and end tokens
        use_same_token = False
        use_molmo_mode = False
    
    # ... token pairing logic based on mode ...
    
    if use_molmo_mode:
        patch_token_id = tokenizer.convert_tokens_to_ids(patch_token)
        im_patch_positions = (input_ids == patch_token_id).nonzero(as_tuple=True)[0]
        
        token_pairs = []
        for start_idx, start_pos in enumerate(start_positions):
            patches_after_start = im_patch_positions[im_patch_positions > start_pos]
            # ... find end of patch sequence for this image ...
            token_pairs.append((start_pos_val, end_pos))
```

---

### 4. Input Parameter Forwarding (`rfm/trainers/rfm_heads_trainer.py`)

**Problem**: Molmo2 uses different input parameter names than Qwen:

| Qwen Parameters | Molmo2 Parameters |
|-----------------|-------------------|
| `image_grid_thw` | `image_grids` |
| `video_grid_thw` | `video_grids` |
| *(none)* | `image_token_pooling` |
| *(none)* | `image_num_crops` |

**Solution**: Pass both sets of parameters, letting each model use what it needs:

```python
model_kwargs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "pixel_values": inputs.get("pixel_values", None),
    "pixel_values_videos": inputs.get("pixel_values_videos", None),
    # Qwen-specific parameters
    "image_grid_thw": inputs.get("image_grid_thw", None),
    "video_grid_thw": inputs.get("video_grid_thw", None),
    "second_per_grid_ts": inputs.get("second_per_grid_ts", None),
    # Molmo2-specific parameters
    "image_grids": inputs.get("image_grids", None),
    "image_token_pooling": inputs.get("image_token_pooling", None),
    "image_num_crops": inputs.get("image_num_crops", None),
    "video_grids": inputs.get("video_grids", None),
    # Common parameters
    "sample_type": sample_type,
    "timing_raw": self.timing_raw,
}
```

---

### 5. Video Processor Attribute Guards (`rfm/models/rfm.py`)

**Problem**: Molmo2's processor doesn't have `temporal_patch_size` or `merge_size` attributes:

```
AttributeError: 'Molmo2VideoProcessor' object has no attribute 'temporal_patch_size'
```

**Solution**: Add `hasattr` checks with defaults:

```python
has_tps = hasattr(self.processor, "video_processor") and \
          hasattr(self.processor.video_processor, "temporal_patch_size")
has_merge = hasattr(self.processor, "video_processor") and \
            hasattr(self.processor.video_processor, "merge_size")

tps = self.processor.video_processor.temporal_patch_size if has_tps else 2
merge_size = self.processor.video_processor.merge_size if has_merge else 14
```

---

## Verification Evidence

Debug logs confirmed the integration is working correctly:

### 1. Embedding Resize
```json
{
  "message": "Starting Molmo2 custom resize (Parameter)",
  "data": {
    "old_num_tokens": 151936,
    "new_vocab_size": 151954,
    "embedding_dim": 2560
  }
}
```

### 2. Collator Output Keys
```json
{
  "message": "Collator batch_inputs keys after processor",
  "data": {
    "batch_keys": ["input_ids", "attention_mask", "token_type_ids", 
                   "pixel_values", "image_token_pooling", "image_grids", 
                   "image_num_crops"],
    "base_model_id": "allenai/Molmo2-4B"
  }
}
```

### 3. Vision Token Detection
```json
{
  "message": "Molmo2 vision positions computed",
  "data": {
    "num_vision_starts": 1,
    "vision_start_positions": [8],
    "num_vision_ends": 1,
    "vision_end_positions": [415],
    "num_im_patches": 392
  }
}
```

### 4. Hidden State Extraction
```json
{
  "message": "Hidden state extraction verification",
  "data": {
    "model_type": "molmo",
    "num_frames_extracted": 1,
    "token_pairs": [[8, 415]],
    "hidden_dim": 2560,
    "embedding_shape": [1, 2560],
    "use_molmo_mode": true
  }
}
```

This confirms:
- ✅ Image starts at position 8 (`<low_res_im_start>`)
- ✅ 392 `<im_patch>` tokens span positions 9-415
- ✅ Hidden states are mean-pooled to shape `[1, 2560]`
- ✅ Molmo2-specific extraction path is used

---

## Usage

Run training with Molmo2:

```bash
uv run train.py \
  trainer_cls=single_frame \
  model.base_model_id=allenai/Molmo2-4B \
  data.use_multi_image=true \
  # ... other args
```

**Note**: Molmo2 only supports multi-image mode (`use_multi_image=true`), not video mode.

---

## Files Modified

| File | Changes |
|------|---------|
| `rfm/utils/setup_utils.py` | Custom embedding resize for Molmo2 |
| `rfm/models/rfm.py` | Vision token handling, hidden state extraction, attribute guards |
| `rfm/trainers/rfm_heads_trainer.py` | Molmo2 input parameter forwarding |
| `rfm/data/collators/rfm_heads.py` | Multi-image mode enforcement for Molmo2 |

