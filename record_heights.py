import os

import pybullet as p

from objects import Object

HEIGHT_FILE = 'heights.csv'


def record_heights():
    if os.path.exists(HEIGHT_FILE):
        os.remove(HEIGHT_FILE)
    object_files = [os.path.join(dp, f) for dp, dn, fn in os.walk("data_models") for f in fn if '.obj' in f]
    p.connect(p.DIRECT)

    for file in object_files:
        object_id = Object.load_object(file, 1)
        aabb = p.getAABB(object_id)
        aabb_min = aabb[0]
        aabb_max = aabb[1]
        min_z = aabb_min[2]
        max_z = aabb_max[2]
        height = max_z - min_z
        with open(HEIGHT_FILE, "a") as f:
            f.write(file + f', {height}\n')


if __name__ == '__main__':
    record_heights()
