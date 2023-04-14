from enum import Enum, auto

import pybullet as p

TABLE_HEIGHT = 1


class Category(Enum):
    # Value of the enum is the size range that the object should be in, relative to the size of a standard table model
    HUMAN = auto(), [.5, 1.5]
    CAT = auto(), [0.1, 0.2]
    DOG = auto(), [0.1, 0.3]
    KITCHEN_TABLE = auto(), [.9, 1.1]
    KITCHEN_CHAIR = auto(), [.9, 1.2]
    STAIRS = auto(), [2, 3]
    SOFA = auto(), [.9, 1.1]
    ARMCHAIR = auto(), [0.9, 1.1]
    BED = auto(), [.3, .5]
    LAMP = auto(), [1, 2]
    DOOR = auto(), [2, 2.5]
    TOILET = auto(), [.7, .9]
    DRESSER = auto(), [.8, 1]
    SHOES = auto(), [.05, .1]
    CARPET = auto(), [0.005, 0.01]
    LAUNDRY_BASKET = auto(), [0.5, 1]
    COFFEE_TABLE = auto(), [0.4, 0.5]
    CAT_TREE = auto(), [0.5, 2]
    TV_STAND = auto(), [0.3, 0.5]
    SHELF = auto(), [1.5, 2.5]
    DESK = auto(), [.9, 1.2]
    DESK_CHAIR = auto(), [1, 1.2]
    TEDDY = auto(), [0.05, 0.15]

    def __init__(self, value, scale):
        self.value = value
        self.scale = scale


class Object:
    def __init__(self, filepath: str, scale: list[int] = (1, 1, 1)):
        split_path = filepath.split('/')
        self.category = split_path[0]
        self.name = split_path[1]
        self.filepath = filepath
        self.scale = scale

    def load(self, loaded_objects):
        if ".obj" in self.filepath:
            visualShapeId = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=self.filepath,
                rgbaColor=None,
                meshScale=self.scale,
            )
            collisionShapeId = p.createCollisionShape(
                shapeType=p.GEOM_MESH, fileName=self.filepath, meshScale=self.scale
            )
            objectId = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collisionShapeId,
                baseVisualShapeIndex=visualShapeId
            )
        else:
            objectId = p.loadURDF(self.filepath)

        loaded_objects[objectId] = self.category

        return objectId, loaded_objects
