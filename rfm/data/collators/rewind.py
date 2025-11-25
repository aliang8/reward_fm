#!/usr/bin/env python3

import numpy as np
import torch

from .rfm_heads import RFMBatchCollator
from .utils import convert_frames_to_pil_images
from rfm.data.dataset_types import PreferenceSample, ProgressSample, SimilaritySample


class ReWiNDBatchCollator(RFMBatchCollator):
    def _process_preference_batch(self, preference_samples: list[PreferenceSample]) -> dict[str, torch.Tensor]:
        """Process a batch of preference samples."""
        # Randomly decide whether chosen trajectory goes first or second
        preference_labels = np.random.randint(0, 2, len(preference_samples))

        if self.load_embeddings:
            # Use embeddings directly from trajectories (already loaded by dataset)
            all_chosen_video_embeddings = []
            all_rejected_video_embeddings = []
            all_chosen_text_embeddings = []
            all_rejected_text_embeddings = []

            for sample in preference_samples:
                # Get embeddings directly from trajectories
                chosen_video_emb = sample.chosen_trajectory.video_embeddings
                chosen_text_emb = sample.chosen_trajectory.text_embedding
                rejected_video_emb = sample.rejected_trajectory.video_embeddings
                rejected_text_emb = sample.rejected_trajectory.text_embedding

                if any(
                    emb is None for emb in [chosen_video_emb, chosen_text_emb, rejected_video_emb, rejected_text_emb]
                ):
                    import ipdb

                    ipdb.set_trace()
                    raise ValueError("Sample trajectories are missing embeddings")

                all_chosen_video_embeddings.append(chosen_video_emb)
                all_chosen_text_embeddings.append(chosen_text_emb)
                all_rejected_video_embeddings.append(rejected_video_emb)
                all_rejected_text_embeddings.append(rejected_text_emb)

            # Stack embeddings into batches
            chosen_video_embeddings = torch.stack(all_chosen_video_embeddings)  # [B, T, D]
            rejected_video_embeddings = torch.stack(all_rejected_video_embeddings)  # [B, T, D]
            chosen_text_embeddings = torch.stack(all_chosen_text_embeddings)  # [B, D]
            rejected_text_embeddings = torch.stack(all_rejected_text_embeddings)  # [B, D]

            # Interleave embeddings based on preference_labels
            frame_len = chosen_video_embeddings.shape[1]
            video_embeddings = torch.empty(len(preference_samples), frame_len * 2, chosen_video_embeddings.shape[2])
            text_embeddings = torch.empty(len(preference_samples), chosen_text_embeddings.shape[1])

            for i in range(len(preference_samples)):
                if preference_labels[i] == 1:  # chosen first
                    video_embeddings[i] = torch.cat([chosen_video_embeddings[i], rejected_video_embeddings[i]], dim=0)
                    text_embeddings[i] = chosen_text_embeddings[i]  # Use chosen text embedding
                else:
                    video_embeddings[i] = torch.cat([rejected_video_embeddings[i], chosen_video_embeddings[i]], dim=0)
                    text_embeddings[i] = chosen_text_embeddings[i]  # Use chosen text embedding

            batch_inputs = {
                "video_embeddings": video_embeddings,  # [B, T*2, D]
                "text_embeddings": text_embeddings,  # [B, D]
            }
        else:
            all_chosen_frames = []
            all_rejected_frames = []
            all_tasks = []

            for i, sample in enumerate(preference_samples):
                # Convert frames to appropriate format using stored shapes
                # NOTE: these should already be padded to max_frames by the data generator
                chosen_frames = convert_frames_to_pil_images(
                    sample.chosen_trajectory.frames, sample.chosen_trajectory.frames_shape
                )
                rejected_frames = convert_frames_to_pil_images(
                    sample.rejected_trajectory.frames, sample.rejected_trajectory.frames_shape
                )
                all_chosen_frames.append(chosen_frames)
                all_rejected_frames.append(rejected_frames)
                all_tasks.append(sample.chosen_trajectory.task)

            frame_len = len(all_chosen_frames[0])
            # [(B*T), C, H, W]
            chosen_video_inputs = self.processor(images=all_chosen_frames, return_tensors="pt")["pixel_values"]
            _, C, H, W = chosen_video_inputs.shape
            chosen_video_inputs = chosen_video_inputs.view(len(preference_samples), frame_len, C, H, W)
            rejected_video_inputs = self.processor(images=all_rejected_frames, return_tensors="pt")["pixel_values"]
            rejected_video_inputs = rejected_video_inputs.view(len(preference_samples), frame_len, C, H, W)

            # interleave them based on preference_labels
            video_inputs = torch.empty(len(preference_samples), frame_len * 2, C, H, W)
            for i in range(len(preference_samples)):
                if preference_labels[i] == 1:  # means chosen first
                    video_inputs[i] = torch.cat([chosen_video_inputs[i], rejected_video_inputs[i]], dim=0)
                else:
                    video_inputs[i] = torch.cat([rejected_video_inputs[i], chosen_video_inputs[i]], dim=0)

            encodings = self.tokenizer(
                all_tasks,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_inputs = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "pixel_values_videos": video_inputs,
            }

        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)
        batch_inputs = self._add_preference_meta(batch_inputs, preference_samples)
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in preference_samples]
        return batch_inputs

    def _process_progress_batch(self, progress_samples: list[ProgressSample]) -> dict[str, torch.Tensor]:
        if self.load_embeddings:
            all_video_embeddings = [sample.trajectory.video_embeddings for sample in progress_samples]
            all_text_embeddings = [sample.trajectory.text_embedding for sample in progress_samples]
            video_embeddings = torch.stack(all_video_embeddings)  # [B, T, D]
            text_embeddings = torch.stack(all_text_embeddings)  # [B, D]

            batch_inputs = {
                "video_embeddings": video_embeddings,  # [B, T, D]
                "text_embeddings": text_embeddings,  # [B, D]
            }
        else:
            all_frames = [
                convert_frames_to_pil_images(sample.trajectory.frames, sample.trajectory.frames_shape)
                for sample in progress_samples
            ]

            # here we directly use dino processor process the images and videos to tensors
            video_inputs = self.processor(images=all_frames, return_tensors="pt")["pixel_values"]
            frame_len = len(all_frames[0])
            _, C, H, W = video_inputs.shape
            video_inputs = video_inputs.view(len(progress_samples), frame_len, C, H, W)

            # here we directly use the tokenizer to process the texts to input_ids and attention_mask
            texts = [sample.trajectory.task for sample in progress_samples]
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            batch_inputs = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "pixel_values_videos": video_inputs,
            }

        if progress_samples[0].trajectory.target_progress is not None:
            batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in progress_samples]
        return batch_inputs

    def _process_similarity_batch(self, similarity_samples: list[SimilaritySample]) -> dict[str, torch.Tensor]:
        """Process a batch of similarity samples.
        
        Batches ref_sim and ref_diff inputs together as [ref_sim_0, ref_diff_0, ref_sim_1, ref_diff_1, ...]
        to reduce memory usage by doing a single forward pass instead of two.
        """
        if self.load_embeddings:
            # Use embeddings directly from trajectories (already loaded by dataset)
            all_ref_video_embeddings = []
            all_ref_text_embeddings = []
            all_sim_video_embeddings = []
            all_diff_video_embeddings = []

            for sample in similarity_samples:
                # Get embeddings directly from trajectories
                ref_video_emb = sample.ref_trajectory.video_embeddings
                ref_text_emb = sample.ref_trajectory.text_embedding
                sim_video_emb = sample.sim_trajectory.video_embeddings
                diff_video_emb = sample.diff_trajectory.video_embeddings

                if any(
                    emb is None
                    for emb in [ref_video_emb, ref_text_emb, sim_video_emb, diff_video_emb]
                ):
                    raise ValueError("Sample trajectories are missing embeddings")

                all_ref_video_embeddings.append(ref_video_emb)
                all_ref_text_embeddings.append(ref_text_emb)
                all_sim_video_embeddings.append(sim_video_emb)
                all_diff_video_embeddings.append(diff_video_emb)

            # Stack embeddings into batches
            ref_video_embeddings = torch.stack(all_ref_video_embeddings)  # [B, T, D]
            ref_text_embeddings = torch.stack(all_ref_text_embeddings)  # [B, D]
            sim_video_embeddings = torch.stack(all_sim_video_embeddings)  # [B, T, D]
            diff_video_embeddings = torch.stack(all_diff_video_embeddings)  # [B, T, D]

            # Create ref_sim inputs (reference vs sim) and ref_diff inputs (reference vs diff)
            frame_len = ref_video_embeddings.shape[1]
            ref_sim_video_embeddings = torch.cat([ref_video_embeddings, sim_video_embeddings], dim=1)  # [B, T*2, D]
            ref_diff_video_embeddings = torch.cat([ref_video_embeddings, diff_video_embeddings], dim=1)  # [B, T*2, D]

            # Interleave ref_sim and ref_diff to create batched inputs: [ref_sim_0, ref_diff_0, ref_sim_1, ref_diff_1, ...]
            batch_size = len(similarity_samples)
            video_embeddings = torch.empty(batch_size * 2, frame_len * 2, ref_video_embeddings.shape[2])
            text_embeddings = torch.empty(batch_size * 2, ref_text_embeddings.shape[1])

            for i in range(batch_size):
                # Even indices: ref_sim
                video_embeddings[i * 2] = ref_sim_video_embeddings[i]
                text_embeddings[i * 2] = ref_text_embeddings[i]
                # Odd indices: ref_diff
                video_embeddings[i * 2 + 1] = ref_diff_video_embeddings[i]
                text_embeddings[i * 2 + 1] = ref_text_embeddings[i]

            batch_inputs = {
                "video_embeddings": video_embeddings,  # [B*2, T*2, D]
                "text_embeddings": text_embeddings,  # [B*2, D]
            }
        else:
            # Process frames
            all_ref_frames = []
            all_sim_frames = []
            all_diff_frames = []
            all_tasks = []

            for sample in similarity_samples:
                # Convert frames to appropriate format using stored shapes
                ref_frames = convert_frames_to_pil_images(
                    sample.ref_trajectory.frames, sample.ref_trajectory.frames_shape
                )
                sim_frames = convert_frames_to_pil_images(
                    sample.sim_trajectory.frames, sample.sim_trajectory.frames_shape
                )
                diff_frames = convert_frames_to_pil_images(
                    sample.diff_trajectory.frames, sample.diff_trajectory.frames_shape
                )

                all_ref_frames.append(ref_frames)
                all_sim_frames.append(sim_frames)
                all_diff_frames.append(diff_frames)
                all_tasks.append(sample.ref_trajectory.task)

            frame_len = len(all_ref_frames[0])

            # Process all frames through processor
            ref_video_inputs = self.processor(images=all_ref_frames, return_tensors="pt")["pixel_values"]
            _, C, H, W = ref_video_inputs.shape
            ref_video_inputs = ref_video_inputs.view(len(similarity_samples), frame_len, C, H, W)

            sim_video_inputs = self.processor(images=all_sim_frames, return_tensors="pt")["pixel_values"]
            sim_video_inputs = sim_video_inputs.view(len(similarity_samples), frame_len, C, H, W)

            diff_video_inputs = self.processor(images=all_diff_frames, return_tensors="pt")["pixel_values"]
            diff_video_inputs = diff_video_inputs.view(len(similarity_samples), frame_len, C, H, W)

            # Concatenate for ref_sim and ref_diff
            ref_sim_video_inputs = torch.cat([ref_video_inputs, sim_video_inputs], dim=1)  # [B, T*2, C, H, W]
            ref_diff_video_inputs = torch.cat([ref_video_inputs, diff_video_inputs], dim=1)  # [B, T*2, C, H, W]

            # Interleave ref_sim and ref_diff to create batched inputs: [ref_sim_0, ref_diff_0, ref_sim_1, ref_diff_1, ...]
            batch_size = len(similarity_samples)
            video_inputs = torch.empty(batch_size * 2, frame_len * 2, C, H, W)
            
            for i in range(batch_size):
                # Even indices: ref_sim
                video_inputs[i * 2] = ref_sim_video_inputs[i]
                # Odd indices: ref_diff
                video_inputs[i * 2 + 1] = ref_diff_video_inputs[i]

            # Process text - duplicate tasks for interleaved batch
            all_tasks_interleaved = []
            for task in all_tasks:
                all_tasks_interleaved.extend([task, task])  # One for ref_sim, one for ref_diff

            encodings = self.tokenizer(
                all_tasks_interleaved,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            batch_inputs = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "pixel_values_videos": video_inputs,
            }

        batch_inputs = self._add_similarity_meta(batch_inputs, similarity_samples)
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in similarity_samples]
        return batch_inputs
