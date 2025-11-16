#!/usr/bin/env python3
"""
Dataset categorization for RFM training.

This module defines categories for datasets to control sampling behavior:
- success: Datasets containing primarily successful trajectories
- failure: Datasets containing failure trajectories
- preference_only: Datasets that should only generate preference samples (no progress/similarity)
- paired: Datasets containing paired human/robot trajectories
"""

DATASET_CATEGORY = {
    "success": [
        # Most datasets are success by default, so this list is intentionally minimal
        # Only explicitly list if needed for special handling
    ],
    "failure": [
        "ykorkmaz_libero_failure_rfm_libero_90_failure",
        "ykorkmaz_libero_failure_rfm_libero_10_failure",
    ],
    "preference_only": [
        "jesbu1_oxe_rfm_oxe_bc_z",
        "jesbu1_oxe_rfm_oxe_dlr_edan_shared_control_converted_externally_to_rlds",
    ],
    "paired": [
        "jesbu1_h2r_rfm_h2r",
        "jesbu1_motif_rfm_motif_rfm",
        "anqil_rh20t_subset_rfm_rh20t_human",
        "anqil_rh20t_subset_rfm_rh20t_robot",
        "jesbu1_ph2d_rfm_ph2d",
    ],
}


def get_preference_only_datasets() -> list[str]:
    """Get list of datasets that should only generate preference samples."""
    return DATASET_CATEGORY.get("preference_only", [])


def get_paired_datasets() -> list[str]:
    """Get list of datasets containing paired human/robot trajectories."""
    return DATASET_CATEGORY.get("paired", [])


def get_failure_datasets() -> list[str]:
    """Get list of datasets containing failure trajectories."""
    return DATASET_CATEGORY.get("failure", [])


def get_success_datasets() -> list[str]:
    """Get list of datasets containing successful trajectories."""
    return DATASET_CATEGORY.get("success", [])


def is_preference_only(dataset_name: str) -> bool:
    """Check if a dataset should only generate preference samples."""
    return dataset_name in DATASET_CATEGORY.get("preference_only", [])


def is_paired(dataset_name: str) -> bool:
    """Check if a dataset contains paired human/robot trajectories."""
    return dataset_name in DATASET_CATEGORY.get("paired", [])


def is_failure(dataset_name: str) -> bool:
    """Check if a dataset contains failure trajectories."""
    return dataset_name in DATASET_CATEGORY.get("failure", [])


def is_success(dataset_name: str) -> bool:
    """Check if a dataset contains successful trajectories."""
    return dataset_name in DATASET_CATEGORY.get("success", [])
