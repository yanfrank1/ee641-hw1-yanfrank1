import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Initialize the dataset.

        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to COCO-style JSON annotations
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform
        # Load and parse annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.images = data['images']
        self.annotations = data['annotations']
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.images}
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)
        self.image_ids = [img['id'] for img in self.images]


    def __len__(self):
        """Return the total number of samples."""
        return len(self.images)
        pass

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, self.image_id_to_filename[image_id])
        image = Image.open(image_path).convert("RGB")
        annotations = self.image_id_to_annotations[image_id]
        boxes = []
        labels = []
        for ann in annotations:
            x1, y1, x2, y2 = ann['bbox']
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0

        target = {
            'boxes': boxes,
            'labels': labels
        }
        return image, target
        pass
