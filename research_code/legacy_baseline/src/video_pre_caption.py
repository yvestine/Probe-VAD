import os
import torch
import json
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import argparse

def load_model():
    """
    Load the model and processor from a given path.
    """
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    float_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=float_dtype,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def get_video_length_and_fps(video_path):
    """
    Get the total duration of a video in seconds and its FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration, fps

@torch.inference_mode()
def infer(model, processor, conversation, temperature=0.1):
    """
    Run inference on a single conversation using the model and processor.
    """
    if not conversation:
        print("Warning: Empty conversation input, skipping inference.")
        return ""
    
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    device = next(model.parameters()).device
    float_dtype = next(model.parameters()).dtype
    
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k == "pixel_values":
                inputs[k] = v.to(device, dtype=float_dtype)
            else:
                inputs[k] = v.to(device)
    
    output_ids = model.generate(**inputs, max_new_tokens=256, temperature=temperature)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

def process_video_in_16frame_steps(video_path, video_stem, model, processor, output_dir, interval):
    """
    1) Step through a video in increments of 16 frames (frame_idx = 0, 16, 32, ...).
    2) Convert that frame index to time center_time = frame_idx / fps.
    3) Define a 10-second window centered around that time [center_time - 5, center_time + 5].
       - Clamp if it goes below 0 or above total_duration.
    4) Dynamically compute fps = 16 / window_duration to sample exactly 16 frames in that interval.
    5) Inference once per segment, store caption in a JSON keyed by the frame index (e.g. "0", "16", ...).
    """
    # Prepare output path and skip if already exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_stem}.json")
    if os.path.isfile(output_path):
        print(f"JSON file already exists for {video_stem}, skipping.")
        return
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found. Skipping.")
        return
    
    # Get duration (in seconds) and fps
    total_duration, original_fps = get_video_length_and_fps(video_path)
    if total_duration == 0:
        print(f"Error: Unable to determine duration for {video_path}. Skipping.")
        return
    
    # Calculate total frames based on original fps
    total_frames = int(total_duration * original_fps)
    
    results = {}
    
    frame_step = 16
    for frame_idx in tqdm(range(0, total_frames, frame_step)):
        # Convert the frame index to a center time
        center_time = frame_idx / original_fps
        
        # Define a 10-second window
        start_time = center_time - interval/2.0
        end_time   = center_time + interval/2.0
        
        # Clamp to [0, total_duration]
        if start_time < 0:
            start_time = 0
        if end_time > total_duration:
            end_time = total_duration
        
        # Compute the actual window duration after clamping
        window_duration = end_time - start_time
        if window_duration <= 0:
            # If for some reason it's 0 or negative, skip
            continue
        
        # Build the conversation for the segment
        conversation = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant analyzing this video segment. "
                    "Summarize the main events or actions in a concise way."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": 2,
                            "start_time": start_time,
                            "end_time": end_time,
                            "max_frames": 10
                        }
                    }
                ]
            }
        ]
        
        # Run single-segment inference
        response = infer(model, processor, conversation, temperature=0.1)
        if not response.strip():
            response = "No detected activity in this segment."
        results[str(frame_idx)] = response
        print(response)
    # Save JSON for this video
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Finished processing {video_stem}. JSON saved to {output_path}")

def process_multiple_videos(index_file, folder_path, model, processor, output_dir="json_results", interval = 10):
    """
    Reads video filenames from an index file. For each video:
      - Video path = folder_path/<videoName>.mp4
      - Process in 16-frame increments (each segment ~10s around that center).
      - Uniformly sample 16 frames from each window by adjusting fps dynamically.
      - Save individual JSON for each video to output_dir.
    """
    with open(index_file, "r") as f:
        video_list = [line.strip().split(" ")[0] for line in f.readlines()]
    for video_name in tqdm(video_list):
        video_path = os.path.join(folder_path, video_name + ".mp4")
        process_video_in_16frame_steps(video_path, video_name, model, processor, output_dir, interval)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process videos using VideoLLaMA3 model')
    parser.add_argument('--video_folder', type=str, default='./data/ucf_crime/videos/',
                       help='Path to the folder containing videos')
    parser.add_argument('--index_file', type=str, default='./data/ucf_crime/annotations/test.txt',
                       help='Path to the index file containing video names')
    parser.add_argument('--output_dir', type=str, default='video_llama3_results',
                       help='Directory to save the JSON results')
    parser.add_argument('--interval', type=int, default=10, help='Time interval to sample every 16 frames')
    
    args = parser.parse_args()
    
    model, processor = load_model()
    process_multiple_videos(args.index_file, args.video_folder, model, processor, args.output_dir, args.interval)
    
    print("All videos processed. Individual JSON files are in:", args.output_dir)
