import math

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data

import objects

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
FAR = 500
NEAR = 0.005
DISTANCE = 100


def setup_simulator(connection_mode: int = pybullet.GUI):
    physics_client = bc.BulletClient(connection_mode=connection_mode, options="--background_color_red=1 "
                                                                              "--background_color_blue=0 "
                                                                              "--background_color_green=0")
    physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to load URDFs
    return physics_client


def set_robot_input_from_keyboard(robot):
    keys = client.getKeyboardEvents()
    turn = 0
    forward = 0
    for k, v in keys.items():

        if (k == client.B3G_RIGHT_ARROW and (v & client.KEY_IS_DOWN)):
            turn = -1
        if (k == client.B3G_LEFT_ARROW and (v & client.KEY_IS_DOWN)):
            turn = 1

        if (k == client.B3G_UP_ARROW and (v & client.KEY_IS_DOWN)):
            forward = 5
        if (k == client.B3G_DOWN_ARROW and (v & client.KEY_IS_DOWN)):
            forward = -5

    robot.applyAction((forward, turn))


def run_simulator(p):
    yaw = 0
    xA, yA, zA = [0, 0, 0]
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
    client = setup_simulator()
    client.setAdditionalSearchPath(
        '/mnt/nvme/object_detection/ShapeNetCore.v2/02946921/adaaccc7f642dee1288ef234853f8b4d/models')  # Set path to load URDFs
    object_id = objects.Object.load_object('model_normalized.obj', 1)
    client.resetBasePositionAndOrientation(object_id, [1, 1, 1], [1, 1, 1, 1])
    while True:
        run_simulator(client)
