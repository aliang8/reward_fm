import torch
from transformers import AutoProcessor, AutoTokenizer

from rfm.data.dataset_types import PreferenceSample, ProgressSample, SampleType, SimilaritySample


class BaseCollator:
    def __init__(
        self,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer = None,
        max_length: int = 1024,
        resized_height: int = 128,
        resized_width: int = 128,
        base_model_id: str = None,
        **kwargs,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.tokenizer = tokenizer
        self.base_model_id = base_model_id

        # Update processor based on base model id
        # if "SmolVLM" in self.base_model_id:
        #     # For image processor
        #     self.processor.image_processor.max_image_size = {"longest_edge": self.resized_height}
        #     self.processor.image_processor.size = {"longest_edge": self.resized_height}
        #     self.processor.image_processor.video_sampling["video_size"] = {"longest_edge": self.resized_height}
    
        #     # for video processor
        #     self.processor.video_processor.max_image_size = {"longest_edge": self.resized_height}
        #     self.processor.video_processor.size = {"longest_edge": self.resized_height}
        #     self.processor.video_processor.video_sampling["video_size"] = {"longest_edge": self.resized_height}

    def __call__(
        self,
        samples: list[SampleType],
    ) -> dict[str, torch.Tensor]:
        """
        Collate a list of samples into separate batches for preferences, progress, and similarities.
        For VQA-based reward modeling, everything goes through language generation.

        Args:
            samples: List of Sample objects or dictionaries that can be converted to Sample objects

        Returns:
            Dictionary containing separate batches for preferences, progress, and similarities
        """
        # Convert dictionaries to Sample objects if needed
        sample_objects = []
        for sample in samples:
            if isinstance(sample, dict):
                # Convert dict to appropriate Sample object based on sample_type
                sample_type = sample.get("sample_type", "unknown")
                if sample_type == "preference":
                    sample_obj = PreferenceSample(**sample)
                elif sample_type == "similarity":
                    sample_obj = SimilaritySample(**sample)
                elif sample_type == "progress":
                    sample_obj = ProgressSample(**sample)
                else:
                    raise ValueError(
                        f"Unknown sample_type: {sample_type}. Must be 'preference', 'similarity', or 'progress'"
                    )
                sample_objects.append(sample_obj)
            elif isinstance(sample, (PreferenceSample, SimilaritySample, ProgressSample)):
                sample_objects.append(sample)
            else:
                raise ValueError(f"Expected Sample object or dict, got {type(sample)}")

        # Separate samples by sample type
        preference_samples = [s for s in sample_objects if s.sample_type == "preference"]
        similarity_samples = [s for s in sample_objects if s.sample_type == "similarity"]
        progress_samples = [s for s in sample_objects if s.sample_type == "progress"]

        # Process preferences
        preference_inputs = {}
        if preference_samples:
            preference_inputs = self._process_preference_batch(preference_samples)

        # Process similarities
        similarity_inputs = {}
        if similarity_samples:
            similarity_inputs = self._process_similarity_batch(similarity_samples)

        # Process progress
        progress_inputs = {}
        if progress_samples:
            progress_inputs = self._process_progress_batch(progress_samples)

        # Return all batches
        return {
            "preference_inputs": preference_inputs,
            "similarity_inputs": similarity_inputs,
            "progress_inputs": progress_inputs,
            "num_preferences": len(preference_samples),
            "num_similarities": len(similarity_samples),
            "num_progress": len(progress_samples),
        }
