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


SYSTEM_PROMPT = """You are a video-to-script synchronization agent.

You are given:

* A video.
* A reference script that explains what is happening in the video.

Your task is to create a timestamped transcript by synchronizing the reference script with the video.

Instructions:

1. Watch and analyze the entire video from beginning to end.
2. Understand what action or event is happening at each point in the video.
3. Match the reference script sentences or phrases to the corresponding visual actions in the video.
4. Use the reference script's wording whenever it correctly describes the visual content.
5. Do not invent narration that is not supported by the reference script.
6. Do not change the meaning of the reference script.
7. Split the reference script into segments according to the visual actions and events.
8. Each segment must be 7 seconds or less.
9. Each segment must correspond to the action or event happening during that time.
10. Start the segment when the corresponding action begins.
11. End the segment when the corresponding action or explanation ends.
12. Keep all segments chronological and non-overlapping.
13. Cover the complete video from start to end.
14. If a sentence is longer than 7 seconds when spoken, split it into smaller segments while preserving its original wording and meaning.
15. Do not assign text to a video section where it does not match the visible content.
16. The "line" will later be converted directly into TTS audio, so keep each line natural and suitable for spoken narration.

Return ONLY valid JSON. Do not return markdown, code fences, explanations, comments, or any text outside the JSON.

Use exactly this schema:

{
"segments": [
{
"line": "<narration text>",
"start": <start time in seconds>,
"end": <end time in seconds>
}
]
}
"""

"""SYSTEM_PROMPT =You are a video script processing agent. You will be given a video and a reference script.
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
