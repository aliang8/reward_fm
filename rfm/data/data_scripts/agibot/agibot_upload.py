import os
import re
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm
import time

from rfm.data.helpers import generate_unique_id
from rfm.data.helpers import create_hf_trajectory
from rfm.data.video_helpers import load_video_frames
from helpers import load_sentence_transformer_model

# Reuse the episode record helper as in the loader
try:
	from rfm.data.data_scripts.agibot import get_episode_record, find_task_json_for_episode
except Exception:
	from .agibot_helper import get_episode_record, find_task_json_for_episode  # type: ignore


_CLIP_FILENAME_RE = re.compile(r"^clip_(\d+)@(.+)\.mp4$")


def _collect_planned_texts(
	planned: List[Tuple[str, int, Dict]],
) -> List[str]:
	"""Collect unique texts (full task and subtasks) for embedding."""
	texts: List[str] = []
	seen: set = set()
	for _video_path, clip_index, episode_rec in planned:
		if clip_index == 0:
			full_text = episode_rec.get("task_name") or episode_rec.get("task_description") or ""
			if full_text and full_text not in seen:
				seen.add(full_text)
				texts.append(full_text)
		else:
			actions = episode_rec.get("label_info", {}).get("action_config", [])
			idx = clip_index - 1
			if 0 <= idx < len(actions) and isinstance(actions[idx], dict):
				text = (actions[idx].get("action_text") or "").strip()
				if text and text not in seen:
					seen.add(text)
					texts.append(text)
	return texts


def _text_for_clip(clip_index: int, episode_rec: Dict) -> Optional[str]:
	"""Return task text for clip_0, or action_text for clip_k (k>=1)."""
	if clip_index == 0:
		return episode_rec.get("task_name") or episode_rec.get("task_description") or None
	actions = episode_rec.get("label_info", {}).get("action_config", [])
	idx = clip_index - 1
	if 0 <= idx < len(actions) and isinstance(actions[idx], dict):
		return (actions[idx].get("action_text") or "").strip() or None
	return None


def _frames_provider_from_file(video_file_path: str):
	"""Return a callable that lazily loads frames from a local video file."""
	def _load():
		try:
			return load_video_frames(video_file_path)
		except Exception:
			return np.empty((0,), dtype=object)
	return _load


def upload_agibotworld_local(
	root_dir: str,
	dataset_label: str = "agibotworld",
	max_frames: int = 64,
	fps: int = 10,
	push_to_hub: bool = False,
	hub_repo_id: Optional[str] = None,
) -> Dataset:
	"""Create a HF Dataset from locally prepared AgiBotWorld clips.

	Expects layout:
	  <root>/<dataset_label>/shard_XXXX/episode_YYYYYY/clip_k@<camera>.mp4

	- clip_0 is the full trajectory -> task_name/task_description
	- clip_1..N map to episode_rec['label_info']['action_config'][k-1]['action_text']
	"""
	base_dir = os.path.join(root_dir, dataset_label)
	if not os.path.isdir(base_dir):
		raise FileNotFoundError(f"Base directory not found: {base_dir}")

	print(f"Scanning local AgiBotWorld folder: {base_dir}")
	scan_start_s = time.time()
	seen_shards: set = set()
	seen_episodes: set = set()
	# Plan all clips (episode records resolved in a second pass)
	planned_raw: List[Tuple[str, int, str]] = []  # (video_file_path, clip_index, episode_id)
	entries: List[Dict] = []
	episodes_scanned = 0

	with os.scandir(base_dir) as shard_iter:
		for shard_entry in shard_iter:
			if not shard_entry.is_dir():
				continue
			shard_name = shard_entry.name
			if not shard_name.startswith("shard_"):
				continue
			shard_dir = shard_entry.path
			seen_shards.add(shard_name)

			with os.scandir(shard_dir) as ep_iter:
				for ep_entry in ep_iter:
					if not ep_entry.is_dir():
						continue
					episode_name = ep_entry.name
					if not episode_name.startswith("episode_"):
						continue
					episode_dir = ep_entry.path

					# Parse episode id
					try:
						episode_id = episode_name.split("episode_")[-1]
					except Exception:
						continue
					seen_episodes.add(str(episode_id))

					with os.scandir(episode_dir) as file_iter:
						for f in file_iter:
							if not f.is_file():
								continue
							m = _CLIP_FILENAME_RE.match(f.name)
							if not m:
								continue
							clip_index = int(m.group(1))
							camera = m.group(2)
							if not camera:
								continue
							planned_raw.append((f.path, clip_index, str(episode_id)))

					# Episode completed (regardless of how many clips matched)
					episodes_scanned += 1
					if episodes_scanned % 200 == 0:
						print(f"Scanned episodes so far: {episodes_scanned}")

	print(
		f"Scan complete in {time.time()-scan_start_s:.1f}s: shards={len(seen_shards)}, episodes={len(seen_episodes)}, clips={len(planned_raw)}"
	)
	if not planned_raw:
		return Dataset.from_dict(
			{
				"id": [],
				"task": [],
				"lang_vector": [],
				"data_source": [],
				"frames": [],
				"is_robot": [],
				"quality_label": [],
				"preference_group_id": [],
				"preference_rank": [],
			}
		)

	# Resolve episode records efficiently: map episode_id -> json_path (via index), then load each JSON once
	print("Resolving episode records...")
	resolve_start_s = time.time()
	unique_episode_ids = sorted(seen_episodes)
	episode_to_json: Dict[str, str] = {}
	for i, eid in enumerate(unique_episode_ids, 1):
		try:
			json_path = find_task_json_for_episode(eid)
		except Exception as e:
			print(f"Warning: no task JSON for episode {eid}: {e}")
			continue
		episode_to_json[eid] = json_path
		if i % 5000 == 0:
			print(f"  indexed {i}/{len(unique_episode_ids)} episodes")

	json_to_records: Dict[str, Dict[str, Dict]] = {}
	for json_path in sorted(set(episode_to_json.values())):
		try:
			with open(json_path, "r", encoding="utf-8") as f:
				entries_list = json.load(f)
			rec_map: Dict[str, Dict] = {}
			for item in entries_list:
				if isinstance(item, dict) and "episode_id" in item:
					rec_map[str(item["episode_id"])] = item
			json_to_records[json_path] = rec_map
		except Exception as e:
			print(f"Warning: failed reading {json_path}: {e}")

	episode_records: Dict[str, Dict] = {}
	for eid, jpath in episode_to_json.items():
		rec = json_to_records.get(jpath, {}).get(str(eid))
		if rec is not None:
			episode_records[eid] = rec

	print(f"Resolved {len(episode_records)}/{len(unique_episode_ids)} episode records in {time.time()-resolve_start_s:.1f}s")

	# Precompute language embeddings for unique texts
	print("Collecting unique texts for embeddings...")
	lang_model = load_sentence_transformer_model()
	texts: List[str] = []
	seen_texts: set = set()
	for _video_path, clip_index, episode_id in planned_raw:
		rec = episode_records.get(episode_id)
		if rec is None:
			continue
		text = _text_for_clip(clip_index, rec)
		if text and text not in seen_texts:
			seen_texts.add(text)
			texts.append(text)
	text_to_vec: Dict[str, np.ndarray] = {}
	print(f"Unique texts: {len(texts)}")
	embed_start_s = time.time()
	if texts:
		vectors = lang_model.encode(texts)
		for t, v in zip(texts, vectors):
			text_to_vec[t] = v
	print(f"Embeddings computed in {time.time()-embed_start_s:.1f}s")

	# Build entries mirroring the loader (use existing video path)
	print(f"Building entries for {len(planned_raw)} clips...")
	for video_file_path, clip_index, episode_id in tqdm(planned_raw, desc="Entries", unit="clip"):
		rec = episode_records.get(episode_id)
		if rec is None:
			continue
		text = _text_for_clip(clip_index, rec)
		if not text:
			continue
		lang_vec = text_to_vec.get(text)
		if lang_vec is None:
			lang_vec = np.zeros((384,), dtype=np.float32)

		traj_dict = {
			"id": generate_unique_id(),
			"frames": _frames_provider_from_file(video_file_path),
			"task": text,
			"is_robot": True,
			"quality_label": "successful",
			"preference_group_id": None,
			"preference_rank": None,
		}

		# Keep the same path; create_hf_trajectory will only write if missing
		entry = create_hf_trajectory(
			traj_dict=traj_dict,
			video_path=video_file_path,
			lang_vector=lang_vec,
			max_frames=max_frames,
			dataset_name=dataset_label,
			use_video=True,
			fps=fps,
		)
		if entry:
			# Replace frames with repo-relative path
			rel_path = os.path.relpath(video_file_path, start=os.path.join(root_dir, dataset_label))
			entry["frames"] = os.path.join(dataset_label, rel_path).replace("\\", "/")
			entries.append(entry)

	print(f"Built {len(entries)} entries")
	ds = Dataset.from_list(entries) if entries else Dataset.from_dict({
		"id": [],
		"task": [],
		"lang_vector": [],
		"data_source": [],
		"frames": [],
		"is_robot": [],
		"quality_label": [],
		"preference_group_id": [],
		"preference_rank": [],
	})

	if push_to_hub and hub_repo_id:
		try:
			ds.push_to_hub(hub_repo_id)
		except Exception as e:
			print(f"Warning: failed to push dataset to hub '{hub_repo_id}': {e}")

	return ds
