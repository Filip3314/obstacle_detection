import os

import pybullet as p

from objects import Object

HEIGHT_FILE = 'heights.csv'


def record_heights():
    """Trawls through all models in data_models and records their heights. All the models we're using have varying dimensions,
    so we're recording those heights so when we spawn them in for data collection we can scale them to sizes that are proportional to
    the other models"""
    if os.path.exists(HEIGHT_FILE):
        os.remove(HEIGHT_FILE)
    object_files = [os.path.join(dp, f) for dp, dn, fn in os.walk("data_models") for f in fn if ('.urdf' in f or '.obj' in f) and not f.startswith('_')]
    p.connect(p.DIRECT)

    for file in object_files:
        object_id = Object.load_object(file, 1)
        aabb = p.getAABB(object_id) # Getting the bounding box
        aabb_min = aabb[0]
        aabb_max = aabb[1]
        min_z = aabb_min[2]
        max_z = aabb_max[2]
        height = max_z - min_z
        with open(HEIGHT_FILE, "a") as f:
            f.write(file + f', {height}\n')


if __name__ == '__main__':
    record_heights()
