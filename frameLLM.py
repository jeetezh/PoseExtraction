"""
Frame-sampling version of expand_script_to_video.py.

Same goal as before - take a video + a PARTIAL script and produce a full
narration transcript covering the entire video - but instead of uploading
the whole video to the model, we sample frames at known timestamps with
ffmpeg and send THOSE to the model. This grounds every segment's start/end
in a real, known timestamp instead of the model's self-reported (and
often unreliable) sense of time.

Why this instead of native video input:
    - The model no longer has to invent timestamps - it picks from a list
      of frame timestamps you already know are correct.
    - We can snap any returned start/end to the nearest actual sampled
      frame as a safety net, which isn't possible with native video input.
    - Cost/context tradeoff: more frames = better timing resolution but
      more tokens per request. Tune --interval accordingly.

Setup:
    pip install openai python-dotenv --break-system-packages
    # requires ffmpeg + ffprobe on PATH
    # requires a running vLLM server, e.g.:
    #   vllm serve Qwen/Qwen3-VL-8B-Instruct \
    #       --limit-mm-per-prompt '{"image": 32, "video": 0}' \
    #       --max-model-len 32768 --port 8000

.env (optional):
    VLLM_BASE_URL=http://localhost:8000/v1
    VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

Usage:
    python expand_script_to_video_frames.py --video video.mp4 --script script.txt
    python expand_script_to_video_frames.py --video video.mp4 --script script.txt --interval 2 --out transcript.json
"""

import os
import re
import sys
import json
import base64
import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


# ---------------------------------------------------------------------------
# Video duration + frame extraction
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    timestamp: float
    path: str


def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_frames(video_path: str, interval_sec: float, out_dir: str, duration: float) -> List[Frame]:
    """Extract one frame every `interval_sec` seconds with ffmpeg."""
    fps = 1.0 / interval_sec
    pattern = os.path.join(out_dir, "frame_%05d.jpg")

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "3", pattern],
        capture_output=True, check=True,
    )

    frame_files = sorted(f for f in os.listdir(out_dir) if f.startswith("frame_"))
    frames = []
    for i, fname in enumerate(frame_files):
        ts = round(i * interval_sec, 2)
        if ts > duration:
            break
        frames.append(Frame(timestamp=ts, path=os.path.join(out_dir, fname)))
    return frames


def encode_frame(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Prompt (adapted to reference known frame timestamps instead of raw video)
# ---------------------------------------------------------------------------

def count_script_sentences(script_text):
    parts = re.split(r"[.!?]+", script_text)
    return len([p for p in parts if p.strip()])


def build_prompt(script_text, video_duration, frame_timestamps):
    n_sentences = count_script_sentences(script_text)
    min_segments = max(n_sentences, int(video_duration // 5))
    ts_list = ", ".join(f"{t:.2f}s" for t in frame_timestamps)

    return f"""You are given a PARTIAL narration script written by the user,
and a sequence of frames sampled from a video at these exact timestamps,
in order:

{ts_list}

The frames are provided in the same order as the list above, so the Nth
image corresponds to the Nth timestamp in that list. The full video is
{video_duration:.1f} seconds long. Your output MUST account for the full
{video_duration:.1f} seconds - do not stop early.

Reference script (existing narration, use where it fits):
\"\"\"
{script_text}
\"\"\"

Do the following:
1. Look at the frames in order and identify every distinct step/moment/
   action shown across them.
2. For parts of the video that the reference script already describes,
   reuse that wording (verbatim or lightly adapted to fit as a segment).
3. For parts of the video that the reference script does NOT cover (skipped
   steps, or the video continuing past where the script ends), WRITE NEW
   narration segments in the same tone/style as the reference script,
   describing what's actually shown in the relevant frame(s).
4. The result must be a single continuous narration that covers the ENTIRE
   video from beginning to end.

CRITICAL TIMESTAMP RULE:
- "start" and "end" for every segment MUST be chosen ONLY from the frame
  timestamp list given above. Do not invent any other numbers. Pick the
  timestamp of the frame where a segment's content begins as its "start",
  and the timestamp of the frame where the NEXT segment begins (or the
  final timestamp / {video_duration:.1f} for the last segment) as its "end".

CRITICAL SEGMENTATION RULES:
- NEVER return the whole script as a single segment. Each segment should
  cover roughly one sentence, clause, or short beat of narration.
- Return at least {min_segments} segments total for this video.
- If frames show content the script never mentions, insert additional
  "generated" segments describing exactly what's shown at that time.
- Segments must jointly span from the first to the last given timestamp.
- Do not invent actions that are not visible in the frames.

Return ONLY valid JSON, no markdown fences, no commentary, in this exact schema:

{{
  "segments": [
    {{"line": "<narration text for this segment>", "start": <one of the given timestamps>, "end": <one of the given timestamps>, "source": "<'script' or 'generated'>"}},
    ...
  ]
}}
"""


# ---------------------------------------------------------------------------
# Call vLLM with multiple frame images instead of raw video
# ---------------------------------------------------------------------------

def expand_via_frames(frames: List[Frame], script_text: str, video_duration: float):
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")

    timestamps = [f.timestamp for f in frames]
    content = []
    for frame in frames:
        content.append({"type": "text", "text": f"Frame at {frame.timestamp:.2f}s:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_frame(frame.path)}"},
        })
    content.append({"type": "text", "text": build_prompt(script_text, video_duration, timestamps)})

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=4096,
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    return parse_model_json(raw)


def parse_model_json(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


# ---------------------------------------------------------------------------
# Safety net: snap any returned start/end to the nearest ACTUAL frame
# timestamp, in case the model drifts from the provided list anyway.
# ---------------------------------------------------------------------------

def snap_to_frame_timestamps(result, frame_timestamps: List[float]):
    def nearest(t):
        return min(frame_timestamps, key=lambda ft: abs(ft - t))

    for seg in result["segments"]:
        seg["start"] = nearest(seg["start"])
        seg["end"] = nearest(seg["end"])
    return result


def validate_segments(result, script_text=None, video_duration=None):
    segments = result["segments"]

    if len(segments) <= 1:
        raise ValueError(
            "Model returned only one segment - it likely echoed the script "
            "instead of expanding it. Try re-running, lower temperature, or "
            "increase --interval so it has more/less frames to reason over."
        )

    if script_text is not None:
        n_sentences = count_script_sentences(script_text)
        if len(segments) < n_sentences:
            print(
                f"Warning: got {len(segments)} segments but the script has "
                f"~{n_sentences} sentences.",
                file=sys.stderr,
            )

    if video_duration is not None:
        last_end = segments[-1]["end"]
        if last_end < video_duration * 0.9:
            print(
                f"Warning: last segment ends at {last_end:.1f}s but video is "
                f"{video_duration:.1f}s - increase --interval density or "
                f"check frame coverage near the end.",
                file=sys.stderr,
            )

    prev_end = 0.0
    for seg in segments:
        if seg["start"] < prev_end - 0.5:
            raise ValueError(f"Out-of-order segment: {seg}")
        prev_end = seg["end"]

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Expand a partial script to cover the full video using sampled frames instead of raw video input."
    )
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--script", required=True, help="Path to your (possibly partial) plain-text script")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between sampled frames")
    parser.add_argument("--out", default=None, help="Optional path to write the JSON transcript to")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.script):
        print(f"Script file not found: {args.script}", file=sys.stderr)
        sys.exit(1)

    with open(args.script, "r", encoding="utf-8") as f:
        script_text = f.read().strip()
    if not script_text:
        print("Script file is empty.", file=sys.stderr)
        sys.exit(1)

    video_duration = get_video_duration(args.video)
    print(f"Video duration: {video_duration:.2f}s", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frames = extract_frames(args.video, args.interval, tmp_dir, video_duration)
        print(f"Sampled {len(frames)} frames every {args.interval}s", file=sys.stderr)
        print(f"Expanding '{args.script}' via {MODEL} @ {BASE_URL} using frame sequence...", file=sys.stderr)

        result = expand_via_frames(frames, script_text, video_duration)
        frame_timestamps = [f.timestamp for f in frames]

    result = snap_to_frame_timestamps(result, frame_timestamps)
    validate_segments(result, script_text=script_text, video_duration=video_duration)

    sources = [seg.get("source", "unknown") for seg in result["segments"]]
    n_script = sources.count("script")
    n_generated = sources.count("generated")
    print(
        f"Segments: {len(result['segments'])} total "
        f"({n_script} from your script, {n_generated} newly generated)",
        file=sys.stderr,
    )
    print(f"Timed to span 0.00s - {result['segments'][-1]['end']:.2f}s (snapped to real frame timestamps)", file=sys.stderr)

    output_json = json.dumps(result, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Wrote transcript to {args.out}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
