import sys
import threading
from itertools import islice

import pybullet as p
import os
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R

from run_simulator import setup_simulator, run_simulator

IMAGE_FOLDER = "data"
RGB_DIR = IMAGE_FOLDER + "/" + "rgb"
DEPTH_DIR = IMAGE_FOLDER + "/" + "depth"
LABEL_FILE = "labels.txt"
LABEL_DIR = IMAGE_FOLDER + "/" + "label"
MODEL_DIR = "models"
IMAGES_PER_MODEL = 10
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
FAR = 500
NEAR = 0.005
DISTANCE = 100


def split_data(data, SIZE):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


class ThreadSafeCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.counter = 0

    def increment(self):
        with self.lock:
            self.counter += 1
            return self.counter


def load_object(object_filepath, client):
    visual_shape_id = client.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=object_filepath,
        rgbaColor=[1, 1, 1, 1],
    )
    object_id = client.createMultiBody(
        baseMass=1.0,
        baseVisualShapeIndex=visual_shape_id,
    )

    return object_id


def linearize_depth(depth_matrix: np.ndarray[float]):
    return (FAR * NEAR / (FAR - (FAR - NEAR) * depth_matrix)).astype(np.uint8)


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

object_files = dict()
categories = {-1}
for dp, dn, fn in os.walk(MODEL_DIR):
    for f in fn:
        if '.obj' in f:
            filepath = os.path.join(dp, f)
            split_dir = filepath.split('/')
            category = int(split_dir[-4])
            categories.add(category)
            object_files[filepath] = category

counter = ThreadSafeCounter()


def gather_data(models):
    client = setup_simulator(p.DIRECT)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0, 0, 0],
        cameraTargetPosition=[DISTANCE, 0, 0],
        cameraUpVector=[0, 0, 1.0],
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=90, aspect=1, nearVal=NEAR, farVal=FAR
    )
    # Capture and label the images
    for obj, obj_category in models.items():
        object_id = load_object(obj, client)
        loaded_object_categories = {-1: -1, object_id: obj_category}
        for i in range(IMAGES_PER_MODEL):
            position = [1, 0, 0]
            orientation = R.random().as_quat()
            client.resetBasePositionAndOrientation(object_id, position, orientation)

            imgs = client.getCameraImage(
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                view_matrix,
                projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )

            # Converting data from camera saveable formats
            rgb = np.reshape(imgs[2], (IMAGE_HEIGHT, IMAGE_WIDTH, 4))
            rgb = rgb.astype(np.uint8)
            depth = np.reshape(imgs[3], [IMAGE_HEIGHT, IMAGE_WIDTH])
            depth = linearize_depth(depth)
            segmentation_list = list(map(lambda pixel_object_id: loaded_object_categories[pixel_object_id], imgs[4]))
            counts = {category: 0 for category in categories}
            for pixel in segmentation_list:
                counts[pixel] += 1
            proportions = {category: (count / (IMAGE_HEIGHT * IMAGE_WIDTH)) for category, count in counts.items()}
            segmentation_map = np.reshape(segmentation_list, [IMAGE_HEIGHT, IMAGE_WIDTH])

            img_number = counter.increment()
            # Saving all of our data
            rgb_image = Image.fromarray(rgb)
            depth_image = Image.fromarray(depth)
            rgb_image.save(f"{RGB_DIR}/{img_number}_rgb.png")
            depth_image.save(f"{DEPTH_DIR}/{img_number}_depth.png")
            np.savetxt(f"{LABEL_DIR}/{img_number}_labels.csv", segmentation_map, fmt='%d', delimiter=',')
            with open(LABEL_FILE, "a") as f:
                f.write(f'{img_number}|{obj_category}|' + str(proportions) + '\n')
        client.removeBody(object_id)

    # Disconnect PyBullet
    client.disconnect()


if __name__ == '__main__':
    num_threads = int(sys.argv[1])
    size = int(len(object_files) / num_threads) + 1
    data_chunks = split_data(object_files, size)
    threads = []
    for data_chunk in data_chunks:
        threads.append(threading.Thread(target=gather_data, args=[data_chunk]))
    for thread in threads:
        thread.start()
