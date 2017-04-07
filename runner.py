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
    output_resolution = "640x480"

    if not os.path.isdir(output_dir):
        os.system("mkdir %s" % (output_dir))

    count = 0
    for entry in os.listdir(input_directory):
        print(entry)
        count += 1
        output_path = "%s/%s.jpg" % (output_directory, count)
        input_path = "%s/%s" % (input_dir, entry.replace(" ", "\ "))
        os.system("convert -resize %s %s %s" % (output_resolution, input_path, output_path))
        print("Successfully converted image: %s" % (count))

def normalize_frame(frame):
    filter_frame = np.zeros([480,640,3])

    for c in range(3):
        max_v = np.max(frame[:, :, c])
        min_v = np.min(frame[:, :, c])
        # filter_frame[:, :, c] = (frame[:, :, c] - min_v) / (max_v - min_v)
        filter_frame[:, :, c] = frame[:, :, c] / 255

    return filter_frame

def pre_process_images(dir, label):
    """
    puts all the images 
    :param dir: directory of images 
    :param label: desired label used for neural network 
    :return: list in form [image, label]
    """
    frames_and_labels = []
    count = 0

    for entry in os.listdir(dir):
        count += 1
        frame = cv2.imread("%s/%s" % (dir, entry))
        frame = normalize_frame(frame)
        frames_and_labels.append([frame, label])
        print("Successfully processed image %s in %s" % (count, dir))

    return frames_and_labels

if __name__ == '__main__':
    converted = True
    if not converted:
        convert_images("images_good_raw", "images_good")
        convert_images("images_bad_raw", "images_bad")

    frames_and_labels = pre_process_images("images_good", 1)
    frames_and_labels += pre_process_images("images_bad", 0)

    print(frames_and_labels[0])
    print(frames_and_labels[-1])

    network = NeuralNetwork(frames_and_labels, (480,640,3), trained=False, epochs=80)