from utils import get_data
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os
import glob
import numpy as np

def viz(ground_truth):
    """
    Create a grid visualization of images with color-coded bounding boxes.
    
    Args:
    - ground_truth [list[dict]]: Ground truth data, where each dict contains:
        - 'boxes': List of bounding boxes [[x_min, y_min, x_max, y_max], ...]
        - 'classes': List of class labels [1, 2, ...]
        - 'filename': Image filename
    """
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Construct the relative path to the images directory
    dir_img = os.path.join(current_dir, 'data', 'images', '*.png')
    
    # Use glob to list PNG files
    file_list = glob.glob(dir_img)
    
    # Extract only the filenames
    file_names = [os.path.basename(file) for file in file_list]
    
    fig, axs = plt.subplots(4, 5, figsize=(20, 10))
        
    for idx, data in enumerate(ground_truth):
        filename = data['filename']
        if filename in file_names:
            x = idx % 4
            y = idx % 5
            # Construct the full path to the image
            img_path = os.path.join(current_dir, 'data', 'images', filename)
            
            # Read the image
            img = Image.open(img_path)
            
            # Select subplot
            axs[x,y].imshow(img)
            
            # Get bounding boxes and classes
            boxes = data['boxes']
            classes = data['classes']
            
            # Draw the boxes
            for box, class_ in zip(boxes, classes):
                y_min, x_min, y_max, x_max = box
                width = x_max - x_min
                height = y_max - y_min
                edge_color = 'red' if class_ == 1 else 'green'
                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=2, edgecolor=edge_color, facecolor='none')
                axs[x,y].add_patch(rect)
            axs[x ,y].axis('off')  
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ground_truth, _ = get_data()
    viz(ground_truth)
