import bpy
from math import radians
from mathutils import Matrix

def set_obj_martix(objname, matrix):
    object = bpy.data.objects[objname]
    matrix = matrix @ Matrix.Rotation(radians(90.0), 4, 'X')
    object.matrix_world = matrix

# object = bpy.context.selected_objects[0]

floor_matrix = Matrix()
floor_matrix[0].xyzw = -4.37114e-008, 1, 4.37114e-008, 0,
floor_matrix[1].xyzw = 0, -8.74228e-008, 2, 0
floor_matrix[2].xyzw = 1, 4.37114e-008, 1.91069e-015, 0
floor_matrix[3].xyzw = 0.0, 0.0, 0.0, 1.0
set_obj_martix('floor', floor_matrix)

floor_matrix = Matrix()
floor_matrix[0].xyzw = -4.37114e-008, 1, 4.37114e-008, 0,
floor_matrix[1].xyzw = 0, -8.74228e-008, 2, 0
floor_matrix[2].xyzw = 1, 4.37114e-008, 1.91069e-015, 0
floor_matrix[3].xyzw = 0.0, 0.0, 0.0, 1.0
set_obj_martix('floor', floor_matrix)