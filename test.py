import pybullet as p
import math
import time
import pybullet_data
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
distance = 100000
start_pos = [0, 0, 1]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
img_w, img_h = 120, 80
p.loadURDF("plane.urdf")
r = p.loadURDF("r2d2.urdf", start_pos, start_orientation)

while True:
    p.stepSimulation()
    robot_position, robot_orientation = p.getBasePositionAndOrientation(r)

    yaw = p.getEulerFromQuaternion(robot_orientation)[-1]
    xA, yA, zA = robot_position
    zA = zA + 0.3 # make the camera a little higher than the robot

    # compute focusing point of the camera
    xB = xA + math.cos(yaw) * distance
    yB = yA + math.sin(yaw) * distance
    zB = zA

    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=[xA, yA, zA],
                        cameraTargetPosition=[xB, yB, zB],
                        cameraUpVector=[0, 0, 1.0]
                    )

    projection_matrix = p.computeProjectionMatrixFOV(
                            fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)

    imgs = p.getCameraImage(img_w, img_h,
                            view_matrix,
                            projection_matrix, shadow=True,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

    time.sleep(1./240.)