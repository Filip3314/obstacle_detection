from enum import Enum

import numpy as np
import pybullet as p

TABLE_HEIGHT = 1


class Category(Enum):
    # Value of the enum is the size range that the object should be in, relative to the size of a standard table model
    NONE = [0, 0, 0]
    HUMAN = [.5, 1.5]
    CAT = [0.1, 0.2]
    DOG = [0.1, 0.3]
    KITCHEN_TABLE = [.9, 1.1]
    KITCHEN_CHAIR = [.9, 1.2]
    STAIRS = [2, 3]
    SOFA = [.9, 1.1]
    ARMCHAIR = [0.9, 1.1]
    BED = [.3, .5]
    LAMP = [1, 2]
    DOOR = [2, 2.5]
    TOILET = [.7, .9]
    DRESSER = [.8, 1]
    SHOES = [.05, .1]
    CARPET = [0.005, 0.01]
    LAUNDRY_BASKET = [0.5, 1]
    COFFEE_TABLE = [0.4, 0.5]
    CAT_TREE = [0.5, 2]
    TV_STAND = [0.3, 0.5]
    SHELF = [1.5, 2.5]
    DESK = [.9, 1.2]
    DESK_CHAIR = [1, 1.2]
    TEDDY = [0.05, 0.15]

    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, scale):
        self.scale = scale


class Object:
    def __init__(self, filepath: str, height: float):
        split_path = filepath.split('/')
        self.category = Category[split_path[1]]
        self.name = split_path[2]
        self.filepath = filepath
        self.scale = np.random.uniform(self.category.scale[0], self.category.scale[1]) / height

    @classmethod
    def load_object(cls, filepath, scale):
        if ".obj" in filepath:
            scale_array = np.repeat(scale, 3)
            visualShapeId = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=filepath,
                rgbaColor=None,
                meshScale=scale_array,
            )
            collisionShapeId = p.createCollisionShape(
                shapeType=p.GEOM_MESH, fileName=filepath, meshScale=scale_array
            )
            objectId = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collisionShapeId,
                baseVisualShapeIndex=visualShapeId
            )
        else:
            objectId = p.loadURDF(filepath, globalScaling=scale)

        return objectId

    def load(self, loaded_objects):
        print(self.filepath)
        object_id = Object.load_object(self.filepath, self.scale)

        loaded_objects[object_id] = self.category.value

        return object_id, loaded_objects
