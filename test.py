import pybullet as p
import math
import pybullet_data
from PIL import Image
import numpy as np

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
distance = 100000
image_count = 0
start_pos = [0, 0, 1]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
img_w, img_h = 240, 160
p.loadSDF("data/kitchens/1.sdf", globalScaling=2)
p.loadURDF("table/table.urdf", [2, 2, 0])
r = p.loadURDF("racecar/racecar_differential.urdf", start_pos, start_orientation)
far = 100
near = 0.02


while True:
    count = 0
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
                            fov=90, aspect=1.5, nearVal=near, farVal=far)

    imgs = p.getCameraImage(img_w, img_h,
                            view_matrix,
                            projection_matrix,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

    count = count + 1
    #if count == 1000
    rgb = np.reshape(imgs[2], (img_h, img_w, 4))
    rgb = rgb.astype(np.uint8)
    depth = np.reshape(imgs[3], [img_h, img_w])
    depth = far * near / (far - (far - near) * depth)
    depth = depth.astype(np.uint8)
#   image = plt.imshow(rgb, interpolation='none', animated=True, label="black")
#   plt.savefig("rgb_{0}".format(image_count))
    rgb_image = Image.fromarray(rgb)
    depth_image = Image.fromarray(depth)
    rgb_image.save('training_data/rgb_{}.png'.format(image_count))
    depth_image.save('training_data/depth_{}.png'.format(image_count))
    image_count = image_count + 1