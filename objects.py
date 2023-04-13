from enum import Enum

Category = Enum(
    "Category", ["table", "cat", "dog", "teddy", "human", "plate", "cup", "pan"]
)


class Object:
    def __init__(self, category: Category, name: str, filepath: str, scale: list[int] = (1, 1, 1)):
        self.category = category
        self.name = name
        self.filepath = filepath
        self.scale = scale

    def load(self, p, position=(1,1,1), orientation=(1, 1, 1, 1)):
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

        return objectId

OBJECTS = [
    Object("table", "table", "table/table.urdf"),
    Object("cat", "cat_2", "cat_2/source/12221_Cat_v1_l3.obj", [.01, .01, .01]),
    #    Object("cat", "cat_1", "cat_1/source/model.obj"),
    #   Object("table", "table_square", "table_square/table_square.urdf"),
    #   Object("human", "human", "humanoid.urdf"),
    #   Object("teddy", "teddy", "teddy_large.urdf"),
    #   Object("plate", "plate", "dinnerware/plate.urdf"),
    #   Object("cup", "cup", "dinnerware/cup/cup_small.urdf"),
    #   Object("pan", "pan", "dinnerware/pan_tefal.urdf"),
    #             Object('dog', 'dog_1/source/*.obj'),
    #             Object('dog', 'dog_2/source/*.obj'),
]
