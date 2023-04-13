from typing import Callable, Optional

import pybullet as p
import pybullet_data
import time
import math
import os
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial.transform import Rotation as R
import pandas as pd
import torch
from torch.utils.data import Dataset
from pybullet_envs.bullet.racecar import Racecar

import objects

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
FAR = 500
NEAR = 0.005
DISTANCE = 100

def setup_simulator():
    p.connect(p.GUI)
    p.setAdditionalSearchPath('pybullet_models')  # Set path to load URDFs
    p.setGravity(0, 0, -9.81)
    return Racecar(p)


def set_robot_input_from_keyboard(robot):
    keys = p.getKeyboardEvents()
    turn = 0
    forward = 0
    for k, v in keys.items():

        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN)):
            turn = -1
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN)):
            turn = 1

        if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
            forward = 5
        if (k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN)):
            forward = -5

    robot.applyAction((forward, turn))


def run_simulator(robot):
    set_robot_input_from_keyboard(robot)
    p.stepSimulation()
    robot_position, robot_orientation = p.getBasePositionAndOrientation(robot.racecarUniqueId)

    yaw = p.getEulerFromQuaternion(robot_orientation)[-1]
    xA, yA, zA = robot_position
    zA = zA + 0.3  # make the camera a little higher than the robot

    # compute focusing point of the camera
    xB = xA + math.cos(yaw) * DISTANCE
    yB = yA + math.sin(yaw) * DISTANCE
    zB = zA

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[xA, yA, zA],
        cameraTargetPosition=[xB, yB, zB],
        cameraUpVector=[0, 0, 1.0],
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=90, aspect=1.5, nearVal=NEAR, farVal=FAR
    )

    imgs = p.getCameraImage(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        view_matrix,
        projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    return imgs

if __name__ == '__main__':
    robot = setup_simulator()
    #p.loadSDF('pybullet_models/kitchens/1.sdf')
    p.loadURDF("plane.urdf")
    objects.OBJECTS[0].load(p)
    objects.OBJECTS[1].load(p)
    while True:
        run_simulator(robot)

