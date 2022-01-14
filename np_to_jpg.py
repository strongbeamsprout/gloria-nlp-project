import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from gloria.datasets.visualization_utils import pyramid_attn_overlay

def np_to_jpg(directory):
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            image = np.load(file)
            if image.shape[0] != 224:
                image = pyramid_attn_overlay(image, (224, 224)).numpy()
            image = ((image - image.min()) / (image.max() - image.min()))
            image = image * 255
            Image.fromarray(image).convert('RGB').save(file.replace('.npy', '.jpg'))

if __name__ == '__main__':
    np_to_jpg('.')
