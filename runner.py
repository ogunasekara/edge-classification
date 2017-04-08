import os
import cv2
import numpy as np

from convnet import NeuralNetwork

def convert_images(input_dir, output_dir):
    """
    converts images from native resolution to 640x480 resolution
    for use in the Convolution Network
    """
    input_directory = input_dir
    output_directory = output_dir
    output_resolution = "320x240"

    if not os.path.isdir(output_dir):
        os.system("mkdir %s" % (output_dir))

    count = 0
    for entry in os.listdir(input_directory):
        count += 1
        output_path = "%s/%s.jpg" % (output_directory, count)
        input_path = "%s/%s" % (input_dir, entry.replace(" ", "\ "))
        os.system("convert -resize %s %s %s" % (output_resolution, input_path, output_path))
        print("Successfully converted image: %s" % (count))

if __name__ == '__main__':
    converted = False
    if not converted:
        convert_images("images_good_raw", "images_good")
        convert_images("images_bad_raw", "images_bad")

    network = NeuralNetwork("images_good", "images_bad", (240,320,3), trained=False, epochs=80)