import logging
logging.basicConfig(level=logging.INFO)

from qwen_vl_utils import process_vision_info

messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": "./video/seibelWorkspace37to97.mp4", "fps": 2.0},
        {"type": "text", "text": "..."}
    ]
}]

image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    return_video_kwargs=True,
    return_video_metadata=True
)

video_tensor, video_metadata = video_inputs[0]   # unpack the tuple correctly
print("video_kwargs:", video_kwargs)
print("video_metadata:", video_metadata)
print(video_tensor.shape[0], "frames extracted")
