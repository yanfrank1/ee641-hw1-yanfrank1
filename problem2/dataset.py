import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap', 
                 heatmap_size=64, sigma=2.0):
        """
        Args:
            image_dir (str): Directory containing images
            annotation_file (str): JSON file mapping image names to keypoints
            output_type (str): 'heatmap' or 'regression'
            heatmap_size (int): Size of heatmap (height=width)
            sigma (float): Gaussian sigma for heatmap
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        images_list = self.annotations["images"]
        self.image_dict = {item["file_name"]: item["keypoints"] for item in images_list}
        self.image_files = list(self.image_dict.keys())
        xs = np.arange(self.heatmap_size)
        ys = np.arange(self.heatmap_size)
        self.xx, self.yy = np.meshgrid(xs, ys, indexing="xy")
        pass
    
    def generate_heatmap(self, keypoints, height, width):
        """
        Generate gaussian heatmaps for keypoints.
        
        Args:
            keypoints: Array of shape [num_keypoints, 2] in (x, y) format
            height, width: Dimensions of the heatmap
            
        Returns:
            heatmaps: Tensor of shape [num_keypoints, height, width]
        """
        # For each keypoint:
        # 1. Create 2D gaussian centered at keypoint location
        # 2. Handle boundary cases
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, height, width), dtype=np.float32)
        scale_x = width / 128.0
        scale_y = height / 128.0
        for i, (x, y) in enumerate(keypoints):
            x_hm = x * scale_x
            y_hm = y * scale_y
            if not (0 <= x_hm < width and 0 <= y_hm < height):
                continue
            heatmaps[i] = np.exp(-((self.xx - x_hm) ** 2 + (self.yy - y_hm) ** 2) / (2 * self.sigma ** 2))

        return torch.from_numpy(heatmaps)
        pass
    
    def __len__(self):
        return len(self.image_files)
        pass
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [1, 128, 128] (grayscale)
            If output_type == 'heatmap':
                targets: Tensor of shape [5, 64, 64] (5 heatmaps)
            If output_type == 'regression':
                targets: Tensor of shape [10] (x,y for 5 keypoints, normalized to [0,1])
        """
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('L')
        image = image.resize((128, 128))
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        keypoints = np.array(self.image_dict[image_name], dtype=np.float32)
        
        if self.output_type == 'heatmap':
            targets = self.generate_heatmap(keypoints, self.heatmap_size, self.heatmap_size)
        else:
            keypoints[:, 0] /= 128.0
            keypoints[:, 1] /= 128.0
            targets = torch.from_numpy(keypoints.flatten())
        return image, targets
        pass
