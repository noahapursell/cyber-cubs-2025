import cv2
import numpy as np
import random
import argparse

def split_and_remove(image_path, probability=0.2, grid_size=15, output_path="output_removed.png"):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            if random.random() < probability:
                image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = 255  # White block

    cv2.imwrite(output_path, image)
    print(f"Processed image saved to {output_path}")
