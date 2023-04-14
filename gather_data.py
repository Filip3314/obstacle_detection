import pybullet as p
import time
import os
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R

from objects import Category
from run_simulator import setup_simulator, run_simulator, NEAR, FAR, IMAGE_HEIGHT, IMAGE_WIDTH

IMAGE_FOLDER = "data"
RGB_DIR = IMAGE_FOLDER + "/" + "rgb"
DEPTH_DIR = IMAGE_FOLDER + "/" + "depth"
LABEL_FILE = "labels.csv"
LABEL_DIR = IMAGE_FOLDER + "/" + "label"


#TODO add randomization of position of objects
#TODO get more data_models

robot = setup_simulator()

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

objects = []
#FIXE LOAD OBJECTS
loaded_objects = {}

# Capture and label the images
for obj in objects:
    object_id, loaded_objects = obj.load(loaded_objects)
    for i in range(2):
        position = [1, 0, 0]
        orientation = R.random().as_quat()
        p.resetBasePositionAndOrientation(object_id, position, orientation)

        imgs = run_simulator(robot)

        filename = f"{obj.name}_{i}"
        rgb = np.reshape(imgs[2], (IMAGE_HEIGHT, IMAGE_WIDTH, 4))
        rgb = rgb.astype(np.uint8)
        depth = np.reshape(imgs[3], [IMAGE_HEIGHT, IMAGE_WIDTH])
        depth = FAR * NEAR / (FAR - (FAR - NEAR) * depth)
        depth = depth.astype(np.uint8)
        segmentation_list = map(lambda pixel_object_id: loaded_objects[pixel_object_id].category, imgs[3])
        counts = {}
        for pixel in segmentation_list:
            counts[Category(pixel)] += 1
        for key, value in enumerate(counts):
            counts[key] = value / (IMAGE_HEIGHT * IMAGE_WIDTH)
        segmentation_map = np.reshape(segmentation_list, [IMAGE_HEIGHT, IMAGE_WIDTH])
        rgb_image = Image.fromarray(rgb)
        depth_image = Image.fromarray(depth)
        np.savetxt(f"{LABEL_DIR}/{filename}_labels.csv", segmentation_map, delimiter=',')
        rgb_image.save(f"{RGB_DIR}/{filename}_rgb.png")
        depth_image.save(f"{DEPTH_DIR}/{filename}_depth.png")
        with open(LABEL_FILE, "a") as f:
            f.write(f"{filename},{obj}\n")
        time.sleep(1)
    p.removeBody(object_id)
    time.sleep(0.1)

# Disconnect PyBullet
p.disconnect()
