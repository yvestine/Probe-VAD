import os
import pickle
import re
from PIL import Image, ImageDraw

def create_video_gif(annotation_dict, video_key, output_gif, fps=5):
    """
    annotation_dict: dict of {video_name: [ [img_path, x1, y1, x2, y2], ... ]}
    video_key: the key corresponding to the single video you want to visualize, e.g. 'Shoplifting010_x264'
    output_gif: string, path where the GIF will be saved (e.g. 'shoplifting010.gif')
    fps: frames per second for the resulting GIF

    This function:
      1) sorts the frames by frame number,
      2) draws bounding boxes,
      3) saves them as a GIF.
    """
    # 1) Extract frames for this video_key
    if video_key not in annotation_dict:
        raise ValueError(f"Video key '{video_key}' not found in the annotation dictionary.")

    frame_annotations = annotation_dict[video_key]

    # 2) Sort the frames by their frame number
    #    Assumes filenames like '000861.jpg', 'image_0861.jpg', etc.
    def get_frame_number(path):
        # Example path: "../ucf_crime/frames/Burglary/Burglary037_x264/001700.jpg"
        # We'll extract the basename "001700.jpg" and parse the integer from it.
        filename = os.path.basename(path)  # "001700.jpg"
        # A simple way: grab all digits from the filename
        match = re.findall(r"\d+", filename)  # e.g. ["001700"]
        if match:
            return int(match[-1])  # e.g. 1700
        else:
            # fallback if we can't find digits
            return 0

    # Sort in ascending order by frame number
    frame_annotations_sorted = sorted(
        frame_annotations,
        key=lambda x: get_frame_number(x[0])  # x[0] is the image_path
    )

    # 3) Generate frames (with bounding boxes) in memory
    frames = []
    for entry in frame_annotations_sorted:
        img_path, x1, y1, x2, y2 = entry
        # Load and draw bounding box
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        frames.append(img)

    # 4) Save frames as a GIF using Pillow
    #    duration is the time in milliseconds per frame => 1000 / fps
    if not frames:
        raise ValueError(f"No frames found for video key {video_key}.")
    duration_ms = int(1000 / fps)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],    # subsequent frames
        duration=duration_ms,
        loop=0                       # 0 = infinite loop
    )
    print(f"GIF saved to {output_gif}")

if __name__ == "__main__":
    # Example usage:
    annotation_pkl = "Test_annotation_modified.pkl"
    video_key = "Abuse028_x264"  # or any other key
    output_gif = "Abuse028_x264.gif"
    
    with open(annotation_pkl, "rb") as f:
        annotation_dict = pickle.load(f)
    
    create_video_gif(annotation_dict, video_key, output_gif, fps=5)
