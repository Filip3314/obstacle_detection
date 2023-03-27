import pybullet as p
import time
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
print("plugin=", plugin)

p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf", [0, 0, -1])
p.loadURDF("r2d2.urdf")

pixelWidth = 320
pixelHeight = 220
camTargetPos = [0, 0, 0]
camDistance = 4
pitch = -10.0
roll = 0
upAxisIndex = 2

while (p.isConnected()):
    start = time.time()
    p.stepSimulation()
    stop = time.time()
    print("stepSimulation %f" % (stop - start))
    p.getBasePositionAndOrientation(

    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll,
                                                        upAxisIndex)
    projectionMatrix = [
        1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
    ]

    start = time.time()
    img_arr = p.getCameraImage(pixelWidth,
                                pixelHeight,
                                viewMatrix=viewMatrix,
                                projectionMatrix=projectionMatrix,
                                shadow=1,
                                lightDirection=[1, 1, 1])
    stop = time.time()
    print("renderImage %f" % (stop - start))

p.unloadPlugin(plugin)
