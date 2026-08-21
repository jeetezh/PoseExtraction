import argparse
import json
import os

import numpy as np
import soundfile as sf
from kokoro import KPipeline

OUT_DIR = "/work/app/audio"

# Cache pipelines per language so we don't reinitialize on every call
_pipelines = {}

DEFAULT_VOICE = "af_heart"
DEFAULT_LANG = "a"
SAMPLE_RATE = 24000


def get_pipeline(lang_code):
    if lang_code not in _pipelines:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


def synthesize(text, voice, lang_code, speed):
    """Run Kokoro once and return a single concatenated float32 audio array."""
    pipeline = get_pipeline(lang_code)
    generator = pipeline(text, voice=voice, speed=speed)
    audio_chunks = []
    for i, (gs, ps, audio) in enumerate(generator):
        audio_chunks.append(audio)
    if not audio_chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(audio_chunks)


def fit_segment_to_duration(
    text,
    target_duration,
    voice=DEFAULT_VOICE,
    lang_code=DEFAULT_LANG,
    sample_rate=SAMPLE_RATE,
    base_speed=0.8,
    tolerance=0.05,
    max_iters=4,
    min_speed=0.5,
    max_speed=1.6,
):
    """
    Generate speech for `text` and iteratively adjust Kokoro's `speed`
    parameter so the resulting audio's duration lands within `tolerance`
    seconds of `target_duration`. Returns (audio, speed_used).
    """
    if target_duration <= 0:
        return np.zeros(0, dtype=np.float32), base_speed

    speed = base_speed
    audio = synthesize(text, voice, lang_code, speed)
    duration = len(audio) / sample_rate

    for _ in range(max_iters):
        if duration <= 0:
            break
        if abs(duration - target_duration) <= tolerance:
            break
        # Kokoro speed scales roughly linearly with output duration, so
        # scale speed by how far off we are and try again.
        speed = speed * (duration / target_duration)
        speed = max(min_speed, min(max_speed, speed))
        new_audio = synthesize(text, voice, lang_code, speed)
        new_duration = len(new_audio) / sample_rate
        audio, duration = new_audio, new_duration
        if speed in (min_speed, max_speed):
            # Hit the speed clamp; further iterating won't help.
            break

    return audio, speed


def pad_to_length(audio, target_samples):
    """Pad the end of `audio` with silence to reach target_samples (no-op if already longer)."""
    if len(audio) >= target_samples:
        return audio
    pad = np.zeros(target_samples - len(audio), dtype=audio.dtype)
    return np.concatenate([audio, pad])


def process_transcript(
    transcript_path,
    output_path,
    voice=DEFAULT_VOICE,
    lang_code=DEFAULT_LANG,
    sample_rate=SAMPLE_RATE,
    base_speed=0.8,
    tolerance=0.05,
):
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]

    # Total track length is driven by the last segment's end time.
    total_duration = max(seg["end"] for seg in segments)
    total_samples = int(round(total_duration * sample_rate))
    track = np.zeros(total_samples, dtype=np.float32)

    for i, seg in enumerate(segments):
        text = seg["line"]
        start = seg["start"]
        end = seg["end"]
        target_duration = end - start

        print(f"[{i}] {start:.2f}-{end:.2f}s (target {target_duration:.2f}s): {text[:60]!r}...")

        audio, used_speed = fit_segment_to_duration(
            text,
            target_duration,
            voice=voice,
            lang_code=lang_code,
            sample_rate=sample_rate,
            base_speed=base_speed,
            tolerance=tolerance,
        )
        actual_duration = len(audio) / sample_rate
        print(
            f"    -> speed={used_speed:.3f}, actual={actual_duration:.2f}s "
            f"(diff={actual_duration - target_duration:+.2f}s)"
        )

        target_samples = int(round(target_duration * sample_rate))
        start_sample = int(round(start * sample_rate))

        if len(audio) <= target_samples:
            # Shorter than (or equal to) the slot: pad with trailing silence
            # so it doesn't drift into the next segment's start time.
            audio = pad_to_length(audio, target_samples)
            end_sample = start_sample + target_samples
        else:
            # Still longer than the slot even after speed adjustment
            # (e.g. speed clamp hit). Don't cut off words — let it
            # overflow into the gap and grow the track if needed.
            print(
                f"    ! segment {i} overflows its slot by "
                f"{actual_duration - target_duration:.2f}s; keeping full audio"
            )
            end_sample = start_sample + len(audio)

        if end_sample > len(track):
            track = pad_to_length(track, end_sample)

        # Overlay (max) rather than overwrite, in case an earlier
        # overflowing segment already wrote into this region.
        track[start_sample:end_sample] = np.maximum(
            track[start_sample:end_sample], audio
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, track, sample_rate)
    print(f"Saved {output_path} ({len(track) / sample_rate:.2f}s)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Render a transcript.json (segments with line/start/end) into a single timed audio track."
    )
    parser.add_argument("transcript", type=str, help="Path to transcript JSON file")
    parser.add_argument(
        "-o",
        "--output",
        default="transcript-output.wav",
        help="Output WAV filename, resolved relative to the current directory unless an absolute path is given",
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"Kokoro voice (default: {DEFAULT_VOICE})")
    parser.add_argument("--lang", default=DEFAULT_LANG, help=f"Language code (default: {DEFAULT_LANG})")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.8,
        help="Starting speed multiplier before duration-fitting kicks in (default: 0.8)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Acceptable duration error in seconds before stopping speed-fit iterations (default: 0.05)",
    )
    args = parser.parse_args()

    output_path = args.output if os.path.isabs(args.output) else os.path.join(".", args.output)

    process_transcript(
        args.transcript,
        output_path,
        voice=args.voice,
        lang_code=args.lang,
        base_speed=args.speed,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
