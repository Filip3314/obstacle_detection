import pybullet as p
import time
import os
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R

import objects
from run_simulator import setup_simulator, run_simulator, NEAR, FAR, IMAGE_HEIGHT, IMAGE_WIDTH

LABEL_FILE = 'labels.csv'
IMAGE_FOLDER = "data"
RGB_DIR = IMAGE_FOLDER + "/" + "rgb"
DEPTH_DIR = IMAGE_FOLDER + "/" + "depth"


#TODO add randomization of position of objects
#TODO get more models
#TODO figure out if we want to generate images with more than one object in them for training/eval

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

with open(LABEL_FILE, "w") as f:
    f.write("filename,category\n")


# Capture and label the images
for obj in objects.OBJECTS:
    objectId = obj.load(p)
    for i in range(2):
        position = [1, 0, 0]
        orientation = R.random().as_quat()
        p.resetBasePositionAndOrientation(objectId, position, orientation)

        imgs = run_simulator(robot)

        filename = f"{obj.name}_{i}"
        rgb = np.reshape(imgs[2], (IMAGE_HEIGHT, IMAGE_WIDTH, 4))
        rgb = rgb.astype(np.uint8)
        depth = np.reshape(imgs[3], [IMAGE_HEIGHT, IMAGE_WIDTH])
        depth = FAR * NEAR / (FAR - (FAR - NEAR) * depth)
        depth = depth.astype(np.uint8)
        rgb_image = Image.fromarray(rgb)
        depth_image = Image.fromarray(depth)
        rgb_image.save(f"{RGB_DIR}/{filename}_rgb.png")
        depth_image.save(f"{DEPTH_DIR}/{filename}_depth.png")
        with open(LABEL_FILE, "a") as f:
            f.write(f"{filename},{obj}\n")
        time.sleep(1)
    p.removeBody(objectId)
    time.sleep(0.1)

# Disconnect PyBullet
p.disconnect()
