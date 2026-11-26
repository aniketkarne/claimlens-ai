"""Qwen VL extractor - vision-based claim extraction via Ollama qwen3-vl:4b."""

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional, Union

import httpx

from app.config import get_settings


QWEN_VL_SYSTEM_PROMPT = """Role:
You are an expert Document Intelligence Agent specializing in extracting structured data from European Motor Vehicle Accident Statements ("Constat Amiable").

Task:
Analyze the provided image of an accident report form. Extract all handwritten and printed text into a structured JSON format.

Form Structure Rules:
1.  Top Section: Contains general accident details (Date, Time, Location, Witnesses).
2.  Left Column (Blue - Vehicle A): Contains details for the first driver/vehicle.
3.  Right Column (Yellow - Vehicle B): Contains details for the second driver/vehicle.
4.  Center Column: Contains the "Circumstances" checkboxes (1-17). Identify which boxes are marked with an 'X' or a checkmark for Column A and Column B.
5.  Bottom Section: Contains the diagram (Section 13) and signatures.

Extraction Instructions:
-   Checkboxes: In Section 12 (Circumstances), list the specific numbers checked for Vehicle A and Vehicle B.
-   Diagram (Section 13): Describe the visual layout. Note the street names, the direction of the arrows for Vehicle A and Vehicle B, and where the red cross (impact point) is located relative to the vehicles.
-   Signatures: Extract the names signed at the bottom.
-   Legibility: If a handwritten field is illegible, return "null" or "illegible". Do not guess.

Output Format:
Return the data in valid JSON format using the following schema:

{
  "accident_details": {
    "date": "string",
    "time": "string",
    "location": "string",
    "injuries": "boolean",
    "property_damage_other": "boolean",
    "witnesses": ["string"]
  },
  "vehicle_a": {
    "insured_name": "string",
    "insured_address": "string",
    "vehicle_make_model": "string",
    "registration_number": "string",
    "insurance_company": "string",
    "policy_number": "string",
    "driver_name": "string",
    "driver_license_number": "string",
    "point_of_impact": "string (e.g., Front, Rear, Left)",
    "visible_damage": "string",
    "remarks": "string"
  },
  "vehicle_b": {
    "insured_name": "string",
    "insured_address": "string",
    "vehicle_make_model": "string",
    "registration_number": "string",
    "insurance_company": "string",
    "policy_number": "string",
    "driver_name": "string",
    "driver_license_number": "string",
    "point_of_impact": "string",
    "visible_damage": "string",
    "remarks": "string"
  },
  "circumstances": {
    "vehicle_a_checked_boxes": [list of integers],
    "vehicle_b_checked_boxes": [list of integers],
    "total_crosses_a": "integer",
    "total_crosses_b": "integer"
  },
  "diagram_analysis": {
    "street_names": ["string"],
    "vehicle_a_direction": "string",
    "vehicle_b_direction": "string",
    "impact_description": "string"
  },
  "signatures": {
    "driver_a_signature": "string",
    "driver_b_signature": "string"
  }
}"""


class QwenVLExtractor:
    """Extracts structured claim data from accident form images using qwen3-vl:4b via Ollama."""

    def __init__(self, model: str = "qwen3-vl:4b"):
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url
        self.model = model

    def extract(self, image_path: Union[str, Path]) -> dict:
        """
        Extract structured claim data from an accident form image.

        Args:
            image_path: Path to the image file (JPG, PNG, etc.)

        Returns:
            Dictionary containing:
            - source: "qwen3-vl:4b"
            - structured_output: The extracted JSON (dict, or None if the
              model output could not be parsed)
            - raw_output: The raw model response (from thinking field)
            - warnings: Any issues encountered
            - latency_seconds: How long the extraction took
            - latency_ms: Same duration in milliseconds (the unit the rest
              of the app and the web UI use)
        """
        import time
        start = time.time()

        warnings = []
        image_path = Path(image_path)

        if not image_path.exists():
            return {
                "source": self.model,
                "structured_output": None,
                "raw_output": None,
                "warnings": [f"Image file not found: {image_path}"],
                "latency_seconds": time.time() - start
            }

        # Read and base64-encode the image
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return {
                "source": self.model,
                "structured_output": None,
                "raw_output": None,
                "warnings": [f"Failed to read image: {e}"],
                "latency_seconds": time.time() - start
            }

        # Call Ollama
        try:
            reply = self._call_ollama_vision(img_b64)
        except Exception as e:
            warnings.append(f"Ollama call failed: {e}")
            latency_ms = round((time.time() - start) * 1000)
            return {
                "source": self.model,
                "structured_output": None,
                "raw_output": str(e),
                "warnings": warnings,
                "latency_seconds": latency_ms / 1000,
                "latency_ms": latency_ms
            }

        thinking = reply.get("thinking", "") or ""
        resp = reply.get("response", "") or ""
        raw_output = thinking if thinking else resp

        # Parse JSON from the response. qwen3-vl puts its visible output in
        # the 'thinking' field, so try that first — but if the thinking field
        # holds only reasoning and the JSON landed in 'response', fall back.
        structured_output = None
        for candidate in (thinking, resp):
            if not candidate:
                continue
            structured_output = self._parse_json(candidate)
            if structured_output is not None:
                break
        if structured_output is None:
            warnings.append("Failed to parse JSON from model output")

        latency_ms = round((time.time() - start) * 1000)
        return {
            "source": self.model,
            "structured_output": structured_output,
            "raw_output": raw_output[:5000],  # truncate raw output
            "warnings": warnings,
            "latency_seconds": latency_ms / 1000,
            "latency_ms": latency_ms
        }

    def _call_ollama_vision(self, img_b64: str) -> dict:
        """
        Call Ollama generate endpoint with image input.

        qwen3-vl:4b puts its visible output in the 'thinking' field, not
        'response', but both are returned so the caller can fall back when
        only one of them carries the JSON.

        Args:
            img_b64: Base64-encoded image (without data URI prefix)

        Returns:
            {"thinking": str, "response": str} with the raw model fields
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": QWEN_VL_SYSTEM_PROMPT,
            "stream": False,
            "images": [img_b64],
            "options": {
                "temperature": 0.1,
                "num_predict": 4096
            }
        }

        # 16k context + thinking can take up to 5 minutes
        timeout = httpx.Timeout(connect=5.0, read=360.0, write=30.0, pool=30.0)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()

            # Output is in 'thinking' field for this model; keep both so the
            # caller can fall back to 'response' when thinking carries no JSON.
            return {
                "thinking": result.get("thinking", "") or "",
                "response": result.get("response", "") or "",
            }

    def _parse_json(self, text: str) -> Optional[dict]:
        """
        Parse a JSON object from the model output text.

        Tries multiple strategies since the model may wrap JSON in markdown
        code blocks or return it as raw text. Only a JSON *object* counts as
        an extraction — a bare list or scalar would crash the caller's
        structured.get(...) chain, so it is rejected here.

        When the text contains several objects (e.g. a schema echo followed
        by the actual answer), the largest parseable one is returned.
        """
        if not text:
            return None

        # Strategy 1: Look for markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Strategy 2: Scan for balanced {...} regions. A first-{..last-}
        # span breaks as soon as trailing prose contains a '}' or the model
        # emits more than one object, so walk the text and parse each
        # balanced object on its own.
        best = None
        for candidate in self._iter_balanced_objects(text):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and (best is None or len(candidate) > len(best[0])):
                best = (candidate, parsed)
        if best is not None:
            return best[1]

        # Strategy 3: Try the whole text as JSON
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        return None

    def _iter_balanced_objects(self, text: str):
        """
        Yield each balanced top-level {...} substring of `text`.

        Brace matching is string-aware: braces inside JSON string values
        (with backslash escapes) are skipped so a value like "curly } brace"
        does not end the object early.
        """
        depth = 0
        start = None
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if depth == 0:
                if ch == "{":
                    start = i
                    depth = 1
                continue
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    start = None
