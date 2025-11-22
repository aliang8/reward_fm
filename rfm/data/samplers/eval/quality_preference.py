from itertools import combinations
from tqdm import tqdm

from rfm.data.samplers.eval.base_pref import BaseQualityPreferenceSampler
from rfm.utils.distributed import rank_0_print


class QualityPreferenceSampler(BaseQualityPreferenceSampler):
    """Dataset that generates preference samples by pairing trajectories with different quality labels for the same task."""

    def __init__(
        self,
        config,
        dataset,
        combined_indices,
        dataset_success_cutoff_map=None,
        is_evaluation=False,
        verbose=True,
        **kwargs,
    ):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)
        
        # Set data_gen_strategy for this sampler
        self.data_gen_strategy = "quality_preference"

        # Generate all possible sample indices upfront (not the actual samples)
        self.sample_indices = self._generate_all_sample_indices()
        rank_0_print(f"Generated {len(self.sample_indices)} quality preference sample indices", verbose=self.verbose)

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible quality preference sample indices (not the actual samples)."""
        sample_indices = []

        # Group trajectories by task and quality label
        task_to_quality_trajs = {}

        rank_0_print(
            f"Generating quality preference samples for {len(self.robot_trajectories)} trajectories",
            verbose=self.verbose,
        )

        for traj_idx in self.robot_trajectories:
            traj = self.dataset[traj_idx]
            task = traj["task"]
            quality_label = traj.get("quality_label", "unknown")

            if task not in task_to_quality_trajs:
                task_to_quality_trajs[task] = {}

            if quality_label not in task_to_quality_trajs[task]:
                task_to_quality_trajs[task][quality_label] = []

            task_to_quality_trajs[task][quality_label].append(traj_idx)

        # Generate pairs for each task
        quality_order = {"failure": 1, "suboptimal": 2, "successful": 3}

        for task in tqdm(task_to_quality_trajs, desc="Generating quality preference samples"):
            quality_groups = task_to_quality_trajs[task]
            quality_labels = list(quality_groups.keys())

            # Only create pairs if we have at least 2 different quality labels
            if len(quality_labels) < 2:
                continue

            # Create pairs of different quality labels
            for quality1, quality2 in combinations(quality_labels, 2):
                trajs1 = quality_groups[quality1]
                trajs2 = quality_groups[quality2]

                if not trajs1 or not trajs2:
                    continue

                # Determine which quality is better (chosen)
                order1 = quality_order.get(quality1, 0)
                order2 = quality_order.get(quality2, 0)

                # Only create pairs if one quality is strictly better than the other
                if order1 > order2:
                    chosen_quality = quality1
                    rejected_quality = quality2
                    chosen_indices = trajs1
                    rejected_indices = trajs2
                elif order2 > order1:
                    chosen_quality = quality2
                    rejected_quality = quality1
                    chosen_indices = trajs2
                    rejected_indices = trajs1
                else:
                    # Same order, skip this pair as we can't reliably compare them
                    continue

                # Create all possible pairs
                for chosen_idx in chosen_indices:
                    for rejected_idx in rejected_indices:
                        sample_indices.append({
                            "chosen_traj_idx": chosen_idx,
                            "rejected_traj_idx": rejected_idx,
                            "task": task,
                            "chosen_quality": chosen_quality,
                            "rejected_quality": rejected_quality,
                        })

        return sample_indices