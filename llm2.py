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

CHANGELOG (sync-drift fixes):
    - redistribute_by_word_count() replaced with fix_segment_timing():
      the old function discarded ALL model timestamps and replaced them
      with a pure word-count-proportional schedule, which is the actual
      cause of audio/video desync. The new version trusts the model's
      video-grounded timestamps and only patches segments that are
      genuinely broken (missing, zero/negative duration, or overlapping).
    - expand_via_vllm() now passes extra_body={"mm_processor_kwargs":
      {"fps": VIDEO_SAMPLE_FPS}} so server-side frame sampling matches
      what you validate locally with qwen_vl_utils.
    - max_tokens raised from 4096 -> 8192 to reduce truncation risk on
      longer videos with many segments.
"""

import os
import re
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

# fps used for server-side video sampling. Keep this in sync with whatever
# you validate locally via qwen_vl_utils.process_vision_info(...).
VIDEO_SAMPLE_FPS = float(os.getenv("VIDEO_SAMPLE_FPS", "2.0"))

# rough estimate of spoken narration pace, used only as a fallback when a
# segment's own timestamps are missing/broken. Tune to your TTS voice.
FALLBACK_WORDS_PER_SEC = float(os.getenv("FALLBACK_WORDS_PER_SEC", "2.5"))


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

def count_script_sentences(script_text):
    parts = re.split(r"[.!?]+", script_text)
    return len([p for p in parts if p.strip()])


def build_prompt(script_text, video_duration):
    n_sentences = count_script_sentences(script_text)
    # rough floor: at least one segment every ~5s of video, or one per
    # script sentence, whichever is larger
    min_segments = max(n_sentences, int(video_duration // 5))

    return f"""You are given a video and a PARTIAL narration script written by
the user. The script may only cover SOME of what happens in the video - it
might be shorter than the video, skip steps, or stop partway through.

The video is exactly {video_duration:.1f} seconds long. Your output MUST
account for the full {video_duration:.1f} seconds - do not stop early.

Reference script (existing narration, use where it fits):
\"\"\"
{script_text}
\"\"\"

Do the following:
1. Watch the ENTIRE video from start to finish and identify every distinct
   step/moment/action shown, second by second if needed.
2. For parts of the video that the reference script already describes,
   reuse that wording (verbatim or lightly adapted to fit as a segment).
3. For parts of the video that the reference script does NOT cover (skipped
   steps, a shorter script than the video needs, or the video simply
   continuing past where the script ends), WRITE NEW narration segments in
   the same tone/style as the reference script, describing what's actually
   shown at that point.
4. The result must be a single continuous narration that covers the ENTIRE
   video from beginning to end, combining reused script content and newly
   generated content seamlessly.

CRITICAL SEGMENTATION RULES - failure to follow these makes the output unusable:
- NEVER return the whole script (or large multi-sentence chunks of it) as a
  single segment. Each segment should cover roughly ONE sentence, clause, or
  short beat of narration - typically 2-8 seconds of video each.
- You MUST return at least {min_segments} segments total for this video.
- If the video shows content, actions, or screens the script never mentions
  (including anything after the point where the script's content ends),
  you MUST insert additional "generated" segments describing exactly what
  is shown on screen at that time. Do not skip over unscripted portions of
  the video - describe them.
- Segments must jointly span from ~0s to ~{video_duration:.1f}s. Do not let
  the last segment's "end" be far short of {video_duration:.1f}s.
- Do not invent actions that are not visible in the video, and do not
  compress multiple distinct on-screen actions into one segment.
- Give "start"/"end" timestamps that reflect what you actually observe in
  the video for each segment - these are used directly for audio sync, so
  they must be as accurate as possible, not just monotonically increasing.

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
  {video_duration:.1f} seconds - do not stop early just because the
  reference script stopped early.
"""


# ---------------------------------------------------------------------------
# Call vLLM (native video input)
# ---------------------------------------------------------------------------

def expand_via_vllm(video_path, script_text, video_duration):
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
                    {"type": "text", "text": build_prompt(script_text, video_duration)},
                ],
            }
        ],
        max_tokens=8192,
        temperature=0.2,
        # Make sure the server samples frames at the same fps you validated
        # locally with qwen_vl_utils.process_vision_info(...). Without this,
        # the server falls back to its own default sampling, which may not
        # match what you tested.
        extra_body={"mm_processor_kwargs": {"fps": VIDEO_SAMPLE_FPS}},
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
# Timing fix: trust the model's video-grounded timestamps, only patch what's
# actually broken. (Replaces the old redistribute_by_word_count(), which
# discarded ALL model timing in favor of a pure word-count schedule - that
# was the main cause of audio/video desync.)
# ---------------------------------------------------------------------------

def fix_segment_timing(result, target_duration, min_seg_seconds=0.5,
                        words_per_sec=FALLBACK_WORDS_PER_SEC,
                        overlap_tolerance=0.25, tail_slack=0.95):
    """
    Repairs only invalid segments (missing timestamps, zero/negative
    duration, or meaningfully overlapping the previous segment) instead of
    re-laying out the entire timeline. Valid model timestamps are left
    untouched, since they reflect what the model actually saw in the video.
    """
    segments = result["segments"]
    n = len(segments)
    if n == 0:
        return result

    prev_end = 0.0
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")

        broken = (
            start is None
            or end is None
            or end <= start
            or start < prev_end - overlap_tolerance
        )

        if broken:
            word_count = max(len(seg["line"].split()), 1)
            est_dur = max(word_count / words_per_sec, min_seg_seconds)
            seg["start"] = round(prev_end, 2)
            seg["end"] = round(prev_end + est_dur, 2)
        else:
            # keep the model's own timestamps, just normalize rounding
            seg["start"] = round(start, 2)
            seg["end"] = round(end, 2)

        prev_end = seg["end"]

    # Only extend the tail if the model stopped noticeably short of the
    # real video duration - don't touch anything else.
    if segments[-1]["end"] < target_duration * tail_slack:
        segments[-1]["end"] = round(target_duration, 2)

    return result


def validate_segments(result, script_text=None, video_duration=None):
    segments = result["segments"]

    if len(segments) <= 1:
        raise ValueError(
            "Model returned only one segment - it likely just echoed the "
            "script instead of expanding it. Try re-running; consider "
            "lowering temperature or strengthening the prompt further."
        )

    if script_text is not None:
        n_sentences = count_script_sentences(script_text)
        if len(segments) < n_sentences:
            print(
                f"Warning: got {len(segments)} segments but the script has "
                f"~{n_sentences} sentences - model may have merged content "
                f"that should've stayed split.",
                file=sys.stderr,
            )

    if video_duration is not None:
        last_end = segments[-1]["end"]
        if last_end < video_duration * 0.9:
            print(
                f"Warning: last segment ends at {last_end:.1f}s but video is "
                f"{video_duration:.1f}s - model likely stopped early and did "
                f"not cover the full video.",
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
    print(f"Expanding '{args.script}' to cover '{args.video}' via {MODEL} @ {BASE_URL} "
          f"(fps={VIDEO_SAMPLE_FPS}) ...", file=sys.stderr)

    result = expand_via_vllm(args.video, script_text, video_duration)
    validate_segments(result, script_text=script_text, video_duration=video_duration)

    sources = [seg.get("source", "unknown") for seg in result["segments"]]
    n_script = sources.count("script")
    n_generated = sources.count("generated")
    print(
        f"Segments: {len(result['segments'])} total "
        f"({n_script} from your script, {n_generated} newly generated)",
        file=sys.stderr,
    )

    result = fix_segment_timing(result, video_duration)
    validate_segments(result, script_text=script_text, video_duration=video_duration)
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
