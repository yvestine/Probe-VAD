import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def visualize_frame_with_bbox(image, bbox, title="Sample"):
    """
    image: a PIL.Image or a Tensor
    bbox: (x1, y1, x2, y2)
    """
    plt.figure()
    # If 'image' is a PIL image, we can show it directly
    plt.imshow(image)
    
    ax = plt.gca()
    x1, y1, x2, y2 = bbox
    
    rect = plt.Rectangle(
        (x1, y1),  # (x, y)
        (x2 - x1), # width
        (y2 - y1), # height
        fill=False,
        linewidth=2
    )
    ax.add_patch(rect)
    
    plt.title(title)
    # Save or plt.show()
    plt.savefig("./box_visualize.pdf")
    plt.close()  # so it doesn't keep popping up windows if running in a loop

class AnomalyDataset(Dataset):
    def __init__(self, annotation_dict, transform=None):
        """
        annotation_dict: dict
            Keys are video names (strings), values are lists of [img_path, x1, y1, x2, y2].
        transform: optional
            A torchvision transform (or any callable) to apply on the loaded image.
        """
        self.samples = []
        self.transform = transform

        # Flatten the (video -> list of bboxes) into a single list
        # Each entry: (img_path, (x1, y1, x2, y2), video_name)
        for video_name, bbox_list in annotation_dict.items():
            for entry in bbox_list:
                img_path, x1, y1, x2, y2 = entry
                # store them as a Python tuple for coords
                self.samples.append((img_path, (x1, y1, x2, y2), video_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, coords, video_name = self.samples[idx]
        print("Loading:", img_path)

        # Load the image with Pillow
        image = Image.open(img_path).convert("RGB")

        # Optional transform (e.g. resize, tensor conversion)
        if self.transform:
            image = self.transform(image)

        return image, coords, img_path, video_name


def anomaly_collate_fn(batch):
    """
    Custom collate function to avoid PyTorch trying to turn bounding boxes into a single tensor.
    batch is a list of tuples: (image, coords, path, video_name)
    """
    images = []
    coords = []
    paths = []
    video_names = []

    for (img, c, p, v) in batch:
        images.append(img)    # Could be a PIL or a torch.Tensor
        coords.append(c)      # This stays as a tuple (x1,y1,x2,y2)
        paths.append(p)
        video_names.append(v)

    return images, coords, paths, video_names


if __name__ == "__main__":
    # Step 1: Load your pickle
    with open("Test_annotation_modified.pkl", "rb") as f:
        annotation_dict = pickle.load(f)

    # Step 2: Create the dataset
    dataset = AnomalyDataset(annotation_dict)

    # Step 3: Create a DataLoader with our custom collate function
    data_loader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=True,
        collate_fn=anomaly_collate_fn
    )

    # Step 4: Visualize a few samples
    for i, (images, coords, paths, video_names) in enumerate(data_loader):
        # Because batch_size=1, images is a list of length 1
        image = images[0]
        bbox = coords[0]  # (x1, y1, x2, y2)
        path = paths[0]
        video_name = video_names[0]

        # Plot the frame + bounding box
        visualize_frame_with_bbox(image, bbox, title=f"{video_name}\n{path}")

        # Limit how many frames we visualize for demonstration
        if i == 4:
            break
