#ROADMAP
#TODO GET MORE MODELS
#TODO SUPPORT POSITION RANDOMIZATION
#TODO SUPPORT MULTIPLE OBJECT SPAWNING
#TODO SUPPORT OBJECT TRANSFORMS
import pybullet as p
import os
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R

from objects import Category, Object
from record_heights import HEIGHT_FILE
from run_simulator import setup_simulator, run_simulator, NEAR, FAR, IMAGE_HEIGHT, IMAGE_WIDTH

IMAGE_FOLDER = "data"
RGB_DIR = IMAGE_FOLDER + "/" + "rgb"
DEPTH_DIR = IMAGE_FOLDER + "/" + "depth"
LABEL_FILE = "labels.txt"
LABEL_DIR = IMAGE_FOLDER + "/" + "label"


def linearize_depth(depth_matrix: np.ndarray[float]):
    return (FAR * NEAR / (FAR - (FAR - NEAR) * depth_matrix)).astype(np.uint8)

robot = setup_simulator(p.DIRECT)

# Define the image folder and label file
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
else:
    files = os.listdir(RGB_DIR)
    for file in files:
        filepath = os.path.join(RGB_DIR, file)
        os.remove(filepath)
    files = os.listdir(DEPTH_DIR)
    for file in files:
        filepath = os.path.join(DEPTH_DIR, file)
        os.remove(filepath)
    files = os.listdir(LABEL_DIR)
    for file in files:
        filepath = os.path.join(LABEL_DIR, file)
        os.remove(filepath)

if os.path.exists(LABEL_FILE):
    os.remove(LABEL_FILE)
heights = np.loadtxt(HEIGHT_FILE, dtype=str, delimiter=',')
objects = [Object(file, float(height)) for file, height in heights]
loaded_object_categories = {-1: 0}
img_number = 0

# Capture and label the images
for obj in objects:
    object_id, loaded_object_categories = obj.load(loaded_object_categories)
    for i in range(2):
        position = [1, 0, 0]
        orientation = R.random().as_quat()
        p.resetBasePositionAndOrientation(object_id, position, orientation)

        imgs = run_simulator(robot)
        filename = f"{obj.name}_{i}"

        #Converting data from camera saveable formats
        rgb = np.reshape(imgs[2], (IMAGE_HEIGHT, IMAGE_WIDTH, 4))
        rgb = rgb.astype(np.uint8)
        depth = np.reshape(imgs[3], [IMAGE_HEIGHT, IMAGE_WIDTH])
        depth = linearize_depth(depth)
        segmentation_list = list(map(lambda pixel_object_id: loaded_object_categories[int(pixel_object_id)], imgs[4]))
        counts = {category.value: 0 for category in Category}
        for pixel in segmentation_list:
            counts[pixel] += 1
        proportions = {category: (count / (IMAGE_HEIGHT * IMAGE_WIDTH)) for category, count in counts.items()}
        segmentation_map = np.reshape(segmentation_list, [IMAGE_HEIGHT, IMAGE_WIDTH])

        #Saving all of our data
        rgb_image = Image.fromarray(rgb)
        depth_image = Image.fromarray(depth)
        rgb_image.save(f"{RGB_DIR}/{img_number}_rgb.png")
        depth_image.save(f"{DEPTH_DIR}/{img_number}_depth.png")
        np.savetxt(f"{LABEL_DIR}/{img_number}_labels.csv", segmentation_map, fmt='%d', delimiter=',')
        with open(LABEL_FILE, "a") as f:
            f.write(str(proportions) + '\n')
        img_number += 1
    p.removeBody(object_id)

# Disconnect PyBullet
p.disconnect()
