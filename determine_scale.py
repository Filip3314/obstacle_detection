import os

from objects import Category

SCALE_FILE = 'object_scale.csv'
object_files = [os.path.join(dp, f) for dp, dn, fn in os.walk("data_models") for f in fn if '.udrf' in f or '.obj' in f]

for file in object_files:


