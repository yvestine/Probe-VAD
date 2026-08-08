import os
import torch
import json
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import numpy as np
import random
import argparse

SEED = 3306
torch.manual_seed(SEED)
np.random.seed(SEED)

random.seed(SEED)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def load_model(model_path, requested_device, attn_implementation):
    model_device = requested_device or device
    if not model_device.startswith("cuda") and attn_implementation == "flash_attention_2":
        print(f"flash_attention_2 requires CUDA; using eager attention on {model_device}")
        attn_implementation = "eager"
    model_kwargs = dict(
        trust_remote_code=True,
        device_map=model_device,
        torch_dtype=torch.bfloat16 if model_device.startswith("cuda") else torch.float32,
    )
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

@torch.inference_mode()
def infer(model, processor, conversation, max_new_tokens):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Move tensors to GPU if available
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(next(model.parameters()).device)

    # If pixel_values exist, ensure they're the correct dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(
            next(model.parameters()).dtype
        )

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

def get_video_fps_and_duration(video_path):
    """
    Returns (fps, duration_in_seconds) of the video at `video_path`.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps == 0:
        return (0, 0)
    duration = frame_count / fps
    return (fps, duration)

def process_suspicious_interval(
    model, processor, video_path, start_frame, end_frame, sample_fps, max_frames,
    max_new_tokens,
):
    """
    Summarizes the suspicious interval (start_frame to end_frame) in the video.
    Converts frames to seconds using actual FPS. Then passes that to the model with a
    prompt indicating this is the suspicious segment, asking for possible crime behaviors.
    """
    fps, _ = get_video_fps_and_duration(video_path)
    if fps <= 0:
        print(f"Warning: FPS=0 for {video_path}, skipping.")
        return ""

    # Convert frame indexes to seconds
    start_sec = start_frame / fps
    end_sec = end_frame / fps

    # Ensure end_sec isn't beyond the actual video length
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    video_duration_sec = total_frames / fps
    end_sec = min(end_sec, video_duration_sec)
    
    conversation = [
        {
            "role": "system",
            "content": "You are an AI assistant analyzing a suspicious segment of a video."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": video_path,
                        "fps": sample_fps,
                        "start_time": start_sec,
                        "end_time": end_sec,
                        "max_frames": max_frames
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Analyze the video interval to identify any possible suspicious behaviors. "
                        "Return your answer strictly as a Python-style list of phrases that could briefly describe "
                        "the suspicious scene splited by commas."
                        "No additional commentary or text, return only the list."
                    )
                }
            ]
        }
    ]

    response = infer(model, processor, conversation, max_new_tokens)
    print(response)
    torch.cuda.empty_cache()
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', default='./data/ucf_crime/videos/')
    parser.add_argument('--index_file', default='./data/ucf_crime/annotations/test.txt')
    parser.add_argument('--intervals_file', default='./data/ucf_crime/scores/rerun_videollama3/highest_lowest_intervals.json')
    parser.add_argument('--output_file', default='./data/ucf_crime/scores/rerun_videollama3/suspicious_part_phrases.json')
    parser.add_argument('--num_jobs', type=int, default=1)
    parser.add_argument('--job_index', type=int, default=0)
    parser.add_argument('--model_path', default='DAMO-NLP-SG/VideoLLaMA3-7B')
    parser.add_argument('--device', default=None)
    parser.add_argument('--attn_implementation', default='flash_attention_2')
    parser.add_argument('--sample_fps', type=float, default=18.0)
    parser.add_argument('--max_frames', type=int, default=180)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    folder_path, index_file = args.video_folder, args.index_file
    intervals_file, output_file = args.intervals_file, args.output_file

    # Load the model and processor
    model, processor = load_model(
        args.model_path, args.device, args.attn_implementation
    )

    # Load intervals => e.g. { "Abuse028_x264": { "interval": [48, 96], "average_score": 0.2 }, ... }
    with open(intervals_file, "r") as f:
        suspicious_intervals = json.load(f)

    # Prepare to store final results
    results = {}
    if args.resume and os.path.isfile(output_file):
        try:
            with open(output_file) as f:
                loaded_results = json.load(f)
            if isinstance(loaded_results, dict):
                results.update(loaded_results)
        except (OSError, json.JSONDecodeError):
            print(f"Warning: ignoring unreadable output checkpoint {output_file}")

    # We read the index file lines. For each line, we have a video filename (w/o extension).
    with open(index_file, "r") as f:
        video_files = [line.split()[0] for line in f.readlines()]
    if args.num_jobs < 1 or not 0 <= args.job_index < args.num_jobs:
        raise ValueError("job_index must be in [0, num_jobs)")
    video_files = list(np.array_split(video_files, args.num_jobs)[args.job_index])

    for filename in tqdm(video_files, desc="Processing suspicious intervals"):
        if args.resume and filename in results:
            print(f"Already summarized {filename}, skipping.")
            continue
        # Example: If line is "Abuse/Abuse028_x264", then the "base" might be "Abuse028_x264"
        # But your intervals are keyed as "Abuse028_x264" (no folder, no .mp4).
        # We'll strip the folder if present, plus .mp4 if present, to match the intervals dict keys.
        base_name = filename.split("/")[-1]  # e.g. "Abuse028_x264"

        # If that base_name is in your intervals file, get the suspicious [start_frame, end_frame].
        if base_name not in suspicious_intervals:
            print(f"No suspicious interval found for {base_name}, skipping.")
            continue

        interval_info = suspicious_intervals[base_name]
        candidates = interval_info.get("candidate_intervals")
        if candidates:
            intervals = [candidate["interval"] for candidate in candidates]
        else:
            # Backward compatibility with the original single-Wmax file.
            intervals = [interval_info["highest_interval"]]

        indexed_name = filename if os.path.splitext(filename)[1] else filename + ".mp4"
        video_path = os.path.join(folder_path, indexed_name)
        if not os.path.exists(video_path):
            video_path = os.path.join(folder_path, base_name + ".mp4")
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found.")
            continue

        # Keep the original prompt and inference function, but run it once for
        # each selected event node instead of once for a fixed Wmax.
        summaries = []
        for start_frame, end_frame in intervals:
            summary = process_suspicious_interval(
                model, processor, video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
                max_new_tokens=args.max_new_tokens,
            )
            if summary:
                summaries.append(summary)
        summary = ", ".join(summaries)

        # Save it
        results[filename] = summary
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        # {
        #     # "start_frame": start_frame,
        #     # "end_frame": end_frame,
        #     # "description": summary
        # }

    # Write all suspicious-part summaries to a JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Done. Saved suspicious part summaries to {output_file}")

if __name__ == "__main__":
    main()
