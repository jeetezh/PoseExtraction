"""
Take a video and a PARTIAL script (shorter than the video, missing steps,
etc.) and produce a full transcript that covers the ENTIRE video - reusing
your script's wording where it fits, and generating new narration to fill
whatever the script doesn't cover.

Setup:
    pip install openai python-dotenv --break-system-packages
    # requires ffprobe (part of ffmpeg) on PATH
    # requires a running vLLM server, e.g.:
    #   VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3-VL-8B-Instruct \
    #       --limit-mm-per-prompt '{"image": 0, "video": 1}' \
    #       --max-model-len 32768 --port 8000

.env (optional):
    VLLM_BASE_URL=http://localhost:8000/v1
    VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

Usage:
    python expand_script_to_video.py --video video.mp4 --script script.txt
    python expand_script_to_video.py --video video.mp4 --script script.txt --out transcript.json
"""

import os
import sys
import json
import base64
import argparse
import subprocess

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


# ---------------------------------------------------------------------------
# Video duration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(script_text):
    return f"""You are given a video and a PARTIAL narration script written by
the user. The script may only cover SOME of what happens in the video - it
might be shorter than the video, skip steps, or stop partway through.

Reference script (existing narration, use where it fits):
\"\"\"
{script_text}
\"\"\"

Do the following:
1. Watch the ENTIRE video from start to finish and identify every distinct
   step/moment/action shown.
2. For parts of the video that the reference script already describes,
   reuse that wording (verbatim or lightly adapted to fit as a segment) -
   do not rewrite content that's already covered well.
3. For parts of the video that the reference script does NOT cover (skipped
   steps, a shorter script than the video needs, or the video simply
   continuing past where the script ends), WRITE NEW narration segments in
   the same tone/style as the reference script, describing what's actually
   shown at that point.
4. The result must be a single continuous narration that covers the ENTIRE
   video from beginning to end, combining reused script content and newly
   generated content seamlessly.

Return ONLY valid JSON, no markdown fences, no commentary, in this exact schema:

{{
  "segments": [
    {{"line": "<narration text for this segment>", "start": <seconds, float>, "end": <seconds, float>, "source": "<'script' or 'generated'>"}},
    ...
  ]
}}

Rules:
- Segments must be in chronological, non-decreasing order.
- "start"/"end" are seconds from the beginning of the video.
- "source" should be "script" if the line came from (or is closely adapted
  from) the reference script, or "generated" if you wrote it to fill a gap
  the script didn't cover.
- Segments together must cover the whole video, from near 0 seconds to near
  the end of the video - do not stop early just because the reference
  script stopped early.
"""


# ---------------------------------------------------------------------------
# Call vLLM (native video input)
# ---------------------------------------------------------------------------

def expand_via_vllm(video_path, script_text):
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")

    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    video_format = os.path.splitext(video_path)[1].lstrip(".").lower() or "mp4"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/{video_format};base64,{video_b64}"},
                    },
                    {"type": "text", "text": build_prompt(script_text)},
                ],
            }
        ],
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
# Word-count time allocation (fixes uneven/wrong model timing)
# ---------------------------------------------------------------------------

def redistribute_by_word_count(result, target_duration, min_seg_seconds=0.8):
    segments = result["segments"]
    n = len(segments)
    if n == 0:
        return result

    word_counts = [max(len(seg["line"].split()), 1) for seg in segments]
    total_words = sum(word_counts)

    raw_durations = [target_duration * (wc / total_words) for wc in word_counts]

    deficit = sum(max(0.0, min_seg_seconds - d) for d in raw_durations)
    if deficit > 0:
        slack_total = sum(max(0.0, d - min_seg_seconds) for d in raw_durations)
        durations = []
        for d in raw_durations:
            if d < min_seg_seconds:
                durations.append(min_seg_seconds)
            else:
                slack = d - min_seg_seconds
                share = (slack / slack_total) * deficit if slack_total > 0 else 0
                durations.append(d - share)
    else:
        durations = raw_durations

    cursor = 0.0
    for seg, dur in zip(segments, durations):
        seg["start"] = round(cursor, 2)
        cursor += dur
        seg["end"] = round(cursor, 2)

    segments[-1]["end"] = round(target_duration, 2)
    return result


def validate_segments(result):
    prev_end = 0.0
    for seg in result["segments"]:
        if seg["start"] < prev_end - 0.5:
            raise ValueError(f"Out-of-order segment: {seg}")
        prev_end = seg["end"]
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Expand a partial script to cover the full video, filling gaps automatically."
    )
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--script", required=True, help="Path to your (possibly partial) plain-text script")
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
    print(f"Expanding '{args.script}' to cover '{args.video}' via {MODEL} @ {BASE_URL} ...", file=sys.stderr)

    result = expand_via_vllm(args.video, script_text)
    validate_segments(result)

    sources = [seg.get("source", "unknown") for seg in result["segments"]]
    n_script = sources.count("script")
    n_generated = sources.count("generated")
    print(f"Segments from your script: {n_script}, newly generated: {n_generated}", file=sys.stderr)

    result = redistribute_by_word_count(result, video_duration)
    validate_segments(result)
    print(f"Timed to span 0.00s - {result['segments'][-1]['end']:.2f}s", file=sys.stderr)

    output_json = json.dumps(result, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Wrote transcript to {args.out}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
