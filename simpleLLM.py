"""
Analyze a video + reference script, produce a timestamped JSON transcript.

Setup:
    pip install openai --break-system-packages
    # requires a running vLLM server, e.g.:
    #   vllm serve Qwen/Qwen3-VL-8B-Instruct --limit-mm-per-prompt '{"video": 1}' --port 8000

Usage:
    python timestamp_script.py --video video.mp4 --script script.txt --out transcript.json
"""

import os
import json
import base64
import argparse
from openai import OpenAI

SYSTEM_PROMPT = """You are a video script processing agent. You will be given a video and a reference script.
Analyze the video and generate a timestamped script that describes what happens, using the reference script's wording where it matches the video.

Rules:
- Each segment must not cross 7 seconds.
- Segments must be in chronological order and cover the full video from start to end.
- Return ONLY valid JSON, no markdown fences, no extra text, in this schema:

{
  "segments": [
    {"line": "<narration text>", "start": <seconds>, "end": <seconds>}
  ]
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.script, "r", encoding="utf-8") as f:
        script_text = f.read().strip()

    with open(args.video, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")
    video_format = os.path.splitext(args.video)[1].lstrip(".").lower() or "mp4"

    client = OpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key="not-needed",
    )

    response = client.chat.completions.create(
        model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/{video_format};base64,{video_b64}"},
                    },
                    {"type": "text", "text": f"Reference script:\n\"\"\"\n{script_text}\n\"\"\""},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)
    output_json = json.dumps(result, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
