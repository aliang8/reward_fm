#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel
from rfm.data.batch_collator import PreferenceSample, SimilaritySample


class BatchPayload(BaseModel):
    samples: List[Union[PreferenceSample, SimilaritySample]]

def post_batch(url: str, payload: Dict[str, Any], timeout_s: float = 120.0) -> Dict[str, Any]:
    """POST a batch payload to the evaluation server and return parsed JSON."""
    resp = requests.post(url.rstrip("/") + "/evaluate_batch", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()
