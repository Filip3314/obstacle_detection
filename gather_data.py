import pybullet as p
import pybullet_data
import time
import math
import os
from PIL import Image
import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial.transform import Rotation as R

Category = Enum(
    "Category", ["table", "cat", "dog", "teddy", "human", "plate", "cup", "pan"]
)


#TODO add randomization of position of objects
#TODO get more models
#TODO figure out if we want to generate images with more than one object in them for training/eval

# Define the categories
@dataclass
class Object:
    category: Category
    name: str
    filepath: str
    scale: list[float] = field(default_factory=lambda: [1, 1, 1])


objects = [
    Object("table", "table", "table/table.urdf"),
    Object("cat", "cat_2", "cat_2/source/12221_Cat_v1_l3.obj", [.01, .01, .01]),
#    Object("cat", "cat_1", "cat_1/source/model.obj"),
#   Object("table", "table_square", "table_square/table_square.urdf"),
#   Object("human", "human", "humanoid.urdf"),
#   Object("teddy", "teddy", "teddy_large.urdf"),
#   Object("plate", "plate", "dinnerware/plate.urdf"),
#   Object("cup", "cup", "dinnerware/cup/cup_small.urdf"),
#   Object("pan", "pan", "dinnerware/pan_tefal.urdf"),
    #             Object('dog', 'dog_1/source/*.obj'),
    #             Object('dog', 'dog_2/source/*.obj'),
]

# Create the PyBullet environment
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to load URDFs
p.setAdditionalSearchPath("models")  # Set path to load URDFs
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("samurai.urdf")

distance = 100
robot_start_pos = [-2, 0, 0.01]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
img_w, img_h = 240, 160
r = p.loadURDF("racecar/racecar_differential.urdf", robot_start_pos, robot_start_orientation)
far = 500
near = 0.005

# Define the image folder and label file
image_folder = "training_data"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
else:
    files = os.listdir(image_folder)
    for file in files:
        filepath = os.path.join(image_folder, file)
        os.remove(filepath)

label_file = "labels.csv"
with open(label_file, "w") as f:
    f.write("filename,category\n")


# Capture and label the images
for obj in objects:
    if ".obj" in obj.filepath:
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj.filepath,
            rgbaColor=None,
            meshScale=obj.scale,
        )
        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH, fileName=obj.filepath, meshScale=obj.scale
        )
        objectId = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId
        )
    else:
        objectId = p.loadURDF(obj.filepath)
    for i in range(2):
        position = [0, 0, 1]
        orientation = R.random().as_quat()
        p.resetBasePositionAndOrientation(objectId, position, orientation)
        p.stepSimulation()
        robot_position, robot_orientation = p.getBasePositionAndOrientation(r)

        yaw = p.getEulerFromQuaternion(robot_orientation)[-1]
        xA, yA, zA = robot_position
        zA = zA + 0.3  # make the camera a little higher than the robot

        # compute focusing point of the camera
        xB = xA + math.cos(yaw) * distance
        yB = yA + math.sin(yaw) * distance
        zB = zA

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[xA, yA, zA],
            cameraTargetPosition=[xB, yB, zB],
            cameraUpVector=[0, 0, 1.0],
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=1.5, nearVal=near, farVal=far
        )

        imgs = p.getCameraImage(
            img_w,
            img_h,
            view_matrix,
            projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        filename = f"{obj.name}_{i}"
        rgb = np.reshape(imgs[2], (img_h, img_w, 4))
        rgb = rgb.astype(np.uint8)
        depth = np.reshape(imgs[3], [img_h, img_w])
        depth = far * near / (far - (far - near) * depth)
        depth = depth.astype(np.uint8)
        rgb_image = Image.fromarray(rgb)
        depth_image = Image.fromarray(depth)
        rgb_image.save(f"{image_folder}/{filename}_rgb.png")
        depth_image.save(f"{image_folder}/{filename}_depth.png")
        with open(label_file, "a") as f:
            f.write(f"{filename},{obj}\n")
        time.sleep(1)
    p.removeBody(objectId)
    time.sleep(0.1)

# Disconnect PyBullet
p.disconnect()
