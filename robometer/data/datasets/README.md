# StrategyFirstDataset Documentation

## Overview

The `StrategyFirstDataset` is a dataset class that first selects a sample type (preference/progress/similarity), then a strategy, then picks a data source uniformly. This is different from other dataset classes that select trajectories first.

**Sampling Flow:**
1. Select sample type (preference/progress/similarity) based on `sample_type_ratio`
2. Select strategy for that sample type based on strategy ratios
3. Filter indices based on strategy requirements
4. Select data source uniformly from available data sources
5. Sample trajectory from selected data source and generate sample

## Strategy Filtering

### REWIND Strategy
- **Requirement**: Only uses **successful trajectories** (`quality_label == "successful"`)
- **Implementation**: Filters indices to only include trajectories from `quality_indices["successful"]`
- **Reason**: We should not rewind suboptimal trajectories; only rewind successful ones to create suboptimal examples

### SUBOPTIMAL Strategy
- **Requirement**: Filters to tasks that have both optimal and suboptimal trajectories (unless RoboArena/RoboReward)
- **RoboArena/RoboReward Exception**: For datasets with `partial_success` (RoboArena/RoboReward), task-level filtering is skipped since they use `partial_success` logic

## Preference Sampling Details

### Chosen Trajectory (Trajectory A)

The chosen trajectory in a preference sample must be:
- **Successful trajectories only** (`quality_label == "successful"`) OR
- **Trajectories with `partial_success = 1.0`** (full success)

**Subsequence Strategy**: Uses `subsample_forward` by default (normal forward sampling)

**Progress & Success Prediction**:
- **Always predict progress and success** if it's trajectory A (chosen trajectory)
- **Exception**: If the chosen trajectory has `partial_success < 1.0`, do **NOT** predict success (only predict progress)

**Partial Success Handling**:
- Can be a `partial_success` trajectory with value < 1.0
- In this case, progress is predicted but success is NOT predicted

### Rejected Trajectory

The rejected trajectory depends on the strategy used:

#### 1. REWIND Strategy
- **Chosen trajectory requirement**: Must be successful (`quality_label == "successful"`)
  - Enforced by `StrategyFirstDataset._filter_indices_by_strategy()` which filters REWIND to only use successful indices
- **Subsequence strategy**: Uses `subsample_rewind` (rewound version of the chosen trajectory)
- **Progress prediction**: ✅ Yes (progress is computed)
- **Success prediction**: ✅ Yes (success labels are computed from progress)

#### 2. DIFFERENT_TASK Strategy
- **Chosen trajectory requirement**: No specific requirement (can be any trajectory)
- **Subsequence strategy**: Uses `subsample_forward` (normal forward sampling)
- **Rejected trajectory**: Any trajectory from a completely different task
- **Progress prediction**: ✅ Yes (but progress is set to `[0.0]` for all timesteps)
- **Success prediction**: ✅ Yes (but success labels are `[0.0]` for all timesteps, computed from progress=0.0)
  - Implementation: `target_progress = [0.0]` → `success_labels = [0.0]` (via `compute_success_labels`)

#### 3. REVERSE_PROGRESS Strategy
- **Chosen trajectory requirement**: Must be successful (`quality_label == "successful"`)
  - **Enforced by**: `StrategyFirstDataset._filter_indices_by_strategy()` filters REVERSE_PROGRESS (for preference samples) to only use successful indices
- **Subsequence strategy**: Uses `subsample_reverse` (reverse version of the chosen trajectory)
- **Progress prediction**: ✅ Yes (progress is computed, but in reverse order)
- **Success prediction**: ✅ Yes (success labels are computed from progress)

#### 4. SUBOPTIMAL Strategy
- **Chosen trajectory requirement**: Must be from a task that has suboptimal trajectories
- **Subsequence strategy**: Uses `subsample_forward` (normal forward sampling)
- **Rejected trajectory**: Suboptimal/failure trajectory from the same task
- **Partial Success Handling**: 
  - For trajectories with `partial_success`, chosen trajectory must have higher `partial_success` than rejected
  - If found trajectory has higher `partial_success`, trajectories are swapped
- **Progress prediction**: ❌ **No** (progress is **masked out** during training)
  - Implementation: `should_compute_progress()` returns `0.0` for trajectories with `quality_label in ["suboptimal", "failure", "failed"]`
  - Progress loss is not computed for suboptimal trajectories (masked via `target_progress_rejected_mask`)
- **Success prediction**: ❌ No (success labels would be `[0.0]` anyway for suboptimal trajectories via `compute_success_labels`)

## Progress Masking

Progress prediction is controlled by the `should_compute_progress()` function in `robometer/data/collators/rbm_heads.py`:

- **Suboptimal/Failure trajectories**: Progress is masked out (`returns 0.0`)
- **Successful trajectories**: Progress is computed (`returns 1.0`)
- **REWIND strategy**: Progress is computed (`returns 1.0`)
- **DIFFERENT_TASK strategy**: Progress is computed (even though it's 0.0)
- **Trajectories with `partial_success`**: Progress is always computed (`returns 1.0`)

The mask is applied in the trainer via `target_progress_chosen_mask` and `target_progress_rejected_mask` tensors.

## Success Label Computation

Success labels are computed from `target_progress` using `compute_success_labels()`:

- If `quality_label in ["failure", "suboptimal", "failed"]`: All success labels are `[0.0]`
- Otherwise: Success labels are `1.0` if `progress >= threshold`, else `0.0`
- For `DIFFERENT_TASK` strategy: Since `target_progress = [0.0]`, all success labels are `[0.0]`

## Strategy Filtering in PrefSampler

The `PrefSampler` handles strategy selection and trajectory pairing:

1. **Non-successful trajectories (non-partial_success)**: Automatically used as rejected with optimal from same task as chosen
2. **Partial success trajectories**: Handled via `_get_different_partial_success_traj()` with swapping logic
3. **Strategy execution**: Handled via `_execute_strategy()` which returns rejected trajectory and subsample strategy

## Progress Sampling Details

### Strategy Requirements

#### 1. FORWARD_PROGRESS (Subsequence)
- **Requirement**: Only uses **successful trajectories** (`quality_label == "successful"`)
- **Enforced by**: `StrategyFirstDataset._filter_indices_by_strategy()` filters FORWARD_PROGRESS (for progress samples) to only use successful indices
- **Progress prediction**: ✅ Yes
- **Success prediction**: ✅ Yes

#### 2. REWIND
- **Requirement**: Only uses **successful trajectories** (`quality_label == "successful"`)
- **Enforced by**: `StrategyFirstDataset._filter_indices_by_strategy()` filters REWIND (for all sample types) to only use successful indices
- **Progress prediction**: ✅ Yes
- **Success prediction**: ✅ Yes

#### 3. DIFFERENT_TASK_INSTRUCTION
- **Requirement**: Can use any trajectory
- **Progress prediction**: ✅ Yes (but progress is set to `[0.0]` for all timesteps)
- **Success prediction**: ✅ Yes (but success labels are set to `[0.0]` for all timesteps)
  - Implementation: `target_progress = [0.0]` → `success_label = [0.0]` (explicitly set in `progress.py`)

#### 4. REVERSE_PROGRESS
- **Requirement**: Only uses **successful trajectories** (`quality_label == "successful"`)
- **Enforced by**: `StrategyFirstDataset._filter_indices_by_strategy()` filters REVERSE_PROGRESS (for progress samples) to only use successful indices
- **Progress prediction**: ✅ Yes (progress computed in reverse order)
- **Success prediction**: ✅ Yes

## Verification Checklist

### Preference Sampling
- ✅ REWIND strategy only uses successful trajectories (enforced in `StrategyFirstDataset`)
- ✅ REVERSE_PROGRESS strategy only uses successful trajectories for preference samples (enforced in `StrategyFirstDataset`)
- ✅ DIFFERENT_TASK sets rejected progress to `[0.0]` (line 351 in `pref.py`)
- ✅ DIFFERENT_TASK success labels are correctly set to `[0.0]` after setting progress to 0.0
- ✅ SUBOPTIMAL progress is masked out (`should_compute_progress` returns 0.0)
- ✅ Chosen trajectory always predicts progress (unless preference-only dataset)
- ✅ Chosen trajectory with `partial_success < 1.0` does NOT predict success

### Progress Sampling
- ✅ FORWARD_PROGRESS strategy only uses successful trajectories (enforced in `StrategyFirstDataset` for progress samples)
- ✅ REWIND strategy only uses successful trajectories (enforced in `StrategyFirstDataset` for all sample types)
- ✅ REVERSE_PROGRESS strategy only uses successful trajectories (enforced in `StrategyFirstDataset` for progress samples)
- ✅ DIFFERENT_TASK_INSTRUCTION sets progress to `[0.0]` (line 157 in `progress.py`)
- ✅ DIFFERENT_TASK_INSTRUCTION success labels are correctly set to `[0.0]` after setting progress to 0.0

## Similarity Sampling Details

Similarity samples compare three trajectories: a reference trajectory (`ref`), a similar trajectory (`sim`), and a different trajectory (`diff`). The model learns to rank `sim` as more similar to `ref` than `diff`.

### Reference Trajectory
- **Always successful**: Must have `quality_label == "successful"` or `partial_success` present
- **Progress prediction**: ✅ Yes (always computed)
- **Success prediction**: ✅ Yes (always computed)

### Strategy Details

#### 1. REWIND Strategy
- **Reference**: Successful trajectory
  - Progress: ✅ Yes
  - Success: ✅ Yes
- **Sim**: Successful trajectory from same task
  - Progress: ✅ Yes
  - Success: ✅ Yes
- **Diff**: Rewound version of reference trajectory
  - Progress: ✅ Yes (progress computed for rewound trajectory)
  - Success: ✅ Yes (success labels computed from progress)

#### 2. SUBOPTIMAL Strategy
- **Reference**: Successful trajectory
  - Progress: ✅ Yes
  - Success: ✅ Yes
- **Sim**: Optimal trajectory from same task
  - Progress: ✅ Yes
  - Success: ✅ Yes
- **Diff**: Suboptimal trajectory from same task
  - Progress: ❌ **No** (progress is **masked out** during training)
  - Success: ✅ Yes (predict 0 - all success labels are 0.0 for suboptimal trajectories)

#### 3. PAIRED_HUMAN_ROBOT Strategy
- **Reference**: Human successful trajectory
  - Progress: ✅ Yes
  - Success: ✅ Yes
- **Sim**: Robot successful trajectory from same task
  - Progress: ✅ Yes
  - Success: ✅ Yes
- **Diff**: Robot trajectory (either suboptimal same task OR different task)
  - **If suboptimal same task**:
    - Progress: ❌ **No** (progress is **masked out** during training)
    - Success: ✅ Yes (predict 0 - all success labels are 0.0 for suboptimal trajectories)
  - **If different task**:
    - Progress: ✅ Yes (but progress is set to `[0.0]` for all timesteps)
    - Success: ✅ Yes (but success labels are `[0.0]` for all timesteps)
    - Implementation: `target_progress = [0.0]` → `success_label = [0.0]` (explicitly set in `sim.py`)

### Similarity Progress Masking

Progress prediction for similarity samples is controlled by the `should_compute_progress()` function in `robometer/data/collators/rbm_heads.py`:

- **Reference trajectory**: Always computes progress (`returns 1.0`)
- **Sim trajectory**: 
  - If optimal/successful: Progress is computed (`returns 1.0`)
  - If suboptimal: Progress is masked out (`returns 0.0`)
- **Diff trajectory**:
  - If suboptimal same task: Progress is masked out (`returns 0.0`)
  - If different task: Progress is computed (even though it's 0.0) (`returns 1.0`)
  - If rewound: Progress is computed (`returns 1.0`)

The mask is applied in the trainer via `target_progress_sim_mask` and `target_progress_diff_mask` tensors.

### Similarity Success Label Computation

Success labels for similarity samples are computed from `target_progress` using `compute_success_labels()`:

- **Reference trajectory**: Success labels computed from progress (1.0 if `progress >= threshold`, else 0.0)
- **Sim trajectory**: 
  - If optimal/successful: Success labels computed from progress
  - If suboptimal: All success labels are `[0.0]` (via `compute_success_labels` for suboptimal trajectories)
- **Diff trajectory**:
  - If suboptimal same task: All success labels are `[0.0]` (via `compute_success_labels` for suboptimal trajectories)
  - If different task: All success labels are `[0.0]` (since `target_progress = [0.0]`)
  - If rewound: Success labels computed from progress

## Verification Checklist

### Preference Sampling
- ✅ REWIND strategy only uses successful trajectories (enforced in `StrategyFirstDataset`)
- ✅ REVERSE_PROGRESS strategy only uses successful trajectories for preference samples (enforced in `StrategyFirstDataset`)
- ✅ DIFFERENT_TASK sets rejected progress to `[0.0]` (line 351 in `pref.py`)
- ✅ DIFFERENT_TASK success labels are correctly set to `[0.0]` after setting progress to 0.0
- ✅ SUBOPTIMAL progress is masked out (`should_compute_progress` returns 0.0)
- ✅ Chosen trajectory always predicts progress (unless preference-only dataset)
- ✅ Chosen trajectory with `partial_success < 1.0` does NOT predict success

### Progress Sampling
- ✅ FORWARD_PROGRESS strategy only uses successful trajectories (enforced in `StrategyFirstDataset` for progress samples)
- ✅ REWIND strategy only uses successful trajectories (enforced in `StrategyFirstDataset` for all sample types)
- ✅ REVERSE_PROGRESS strategy only uses successful trajectories (enforced in `StrategyFirstDataset` for progress samples)
- ✅ DIFFERENT_TASK_INSTRUCTION sets progress to `[0.0]` (line 157 in `progress.py`)
- ✅ DIFFERENT_TASK_INSTRUCTION success labels are correctly set to `[0.0]` after setting progress to 0.0

### Similarity Sampling
- ✅ REWIND strategy: Reference is successful, sim is successful same task, diff is rewound
- ✅ SUBOPTIMAL strategy: Reference is successful, sim is optimal same task, diff is suboptimal same task (progress masked, success=0)
- ✅ PAIRED_HUMAN_ROBOT strategy: Reference is human successful, sim is robot successful same task, diff is robot suboptimal same task OR different task
- ✅ Different task trajectories have progress=0 and success=0 explicitly set (in `sim.py`)
- ✅ Suboptimal trajectories have progress masked out (`should_compute_progress` returns 0.0)

