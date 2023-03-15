import os
from os import path
from math import pi, degrees, cos, tan, atan
import xml.etree.cElementTree as ET

import bpy
import bmesh
from mathutils import Matrix

def separate_triangulate_mesh(object):
    object.select_set(True)
    # separate by material
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='MATERIAL')
    bpy.ops.object.mode_set(mode='OBJECT')
    # triangulate
    me = object.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    object.select_set(False)

def preprocess_mesh():
    # deselect all
    for object in bpy.context.selected_objects:
        object.select_set(False)

    # gen single material triangles
    for object in bpy.context.view_layer.objects:
        if object.type == 'MESH':
            separate_triangulate_mesh(object)
         # to_do: object.to_mesh()

def get_mesh_ply_path(object, out_dir='', is_absolute=False):
    name = bpy.path.clean_name(object.name_full)
    if is_absolute:
        return f'{out_dir}meshes/{name}.obj'.lower()
    else:
        return f'meshes/{name}.obj'.lower()
         
def export_mesh_ply(out_dir):
    # refresh
    bpy.context.view_layer.update()
    for object in bpy.context.selected_objects:
        object.select_set(False)
    
    # export ply
    if not path.exists(out_dir + 'meshes'):
        os.mkdir(out_dir + 'meshes')
    for object in bpy.context.view_layer.objects:
        if object.type == 'MESH':
            object.select_set(True)
            try:
                bpy.ops.export_mesh.ply(filepath=get_mesh_ply_path(object, out_dir, True), use_ascii=True, use_selection=True, use_colors=False)
            except:
                print(f"export_mesh faliure {get_mesh_ply_path(object, out_dir, True)}")
            object.select_set(False)

def export_mesh(out_dir):
    # refresh
    bpy.context.view_layer.update()
    for object in bpy.context.selected_objects:
        object.select_set(False)

    # export obj
    if not path.exists(out_dir + 'meshes'):
        os.mkdir(out_dir + 'meshes')
    for object in bpy.context.view_layer.objects:
        if object.type in {'MESH', 'CURVE', 'FONT', 'SURFACE'}:
            object.select_set(True)
            try:
                bpy.ops.export_scene.obj(
                    filepath=get_mesh_ply_path(object, out_dir, True),
                    use_selection=True, 
                    use_mesh_modifiers=True,
                    use_materials=False,
                    use_triangles=True,
                    group_by_material=True,
                    axis_forward='Y', 
                    axis_up='Z')
            except:
                print(f"export mesh failed. {get_mesh_ply_path(object, out_dir, True)}")
            object.select_set(False)

def write_xml_materials(xml_scene):
    for material in bpy.data.materials:
        if material.name_full == 'Dots Stroke':
            continue
        xml_material = ET.SubElement(xml_scene, 'material')
        xml_material.set('name', material.name_full)
        if material.use_nodes:
            output_node = material.node_tree.nodes['Material Output']
            surface_node = output_node.inputs["Surface"].links[0].from_node
            if surface_node.type =='BSDF_DIFFUSE':
                xml_material.set('type', 'diffuse')
                xml_property = ET.SubElement(xml_material, 'property')

                color_socket = surface_node.inputs['Color']
                if not color_socket.is_linked:
                    color = color_socket.default_value
                    xml_property.set('color', f'{color[0]} {color[1]} {color[2]}')
                else:
                    texture_node = surface_node.inputs["Color"].links[0].from_node
                    if texture_node.type =='TEX_IMAGE':
                        xml_property.set('colorTexture', texture_node.image.filepath)
            elif surface_node.type =='BSDF_GLOSSY':
                xml_material.set('type', 'metal')
                xml_property = ET.SubElement(xml_material, 'property')

                color_socket = surface_node.inputs['Color']
                if not color_socket.is_linked:
                    color = color_socket.default_value
                    xml_property.set('color', f'{color[0]} {color[1]} {color[2]}')
                else:
                    xml_property.set('color', '1.0 1.0 1.0')
                
                roughness = surface_node.inputs['Roughness'].default_value
                xml_property.set('roughness', str(roughness))
            elif surface_node.type =='BSDF_GLASS' or surface_node.type =='BSDF_REFRACTION':
                xml_material.set('type', 'glass')
                xml_property = ET.SubElement(xml_material, 'property')

                color_socket = surface_node.inputs['Color']
                if not color_socket.is_linked:
                    color = color_socket.default_value
                    xml_property.set('color', f'{color[0]} {color[1]} {color[2]}')
                else:
                    xml_property.set('color', '1.0 1.0 1.0')

                roughness = surface_node.inputs['Roughness'].default_value
                xml_property.set('roughness', str(roughness))

                ior = surface_node.inputs['IOR'].default_value
                xml_property.set('ior', str(ior))
            elif surface_node.type =='BSDF_PRINCIPLED':
                xml_material.set('type', 'principled')
                xml_property = ET.SubElement(xml_material, 'property')
                
                basecolor_socket = surface_node.inputs['Base Color']
                if not basecolor_socket.is_linked:
                    color = basecolor_socket.default_value
                    xml_property.set('baseColor', f'{color[0]} {color[1]} {color[2]}')
                else:
                    xml_property.set('baseColor', '1.0 1.0 1.0')
                
                metallic = surface_node.inputs['Metallic'].default_value
                xml_property.set('metallic', str(metallic))

                roughness = surface_node.inputs['Roughness'].default_value
                xml_property.set('roughness', str(roughness))

def write_xml_transform(xml_node, object):
    m = object.matrix_world
    martix_str = f'{m[0][0]} {m[0][1]} {m[0][2]} {m[0][3]} '
    martix_str += f'{m[1][0]} {m[1][1]} {m[1][2]} {m[1][3]} '
    martix_str += f'{m[2][0]} {m[2][1]} {m[2][2]} {m[2][3]} '
    martix_str += f'{m[3][0]} {m[3][1]} {m[3][2]} {m[3][3]}'

    xml_transform = ET.SubElement(xml_node, 'transform')
    xml_transform.set('matrix', martix_str)

def write_xml_camera(xml_scene, object):
    camera = object.data
    xml_camera = ET.SubElement(xml_scene, 'camera')
    xml_property = ET.SubElement(xml_camera, 'property')
    if camera.type == 'PERSP':
        xml_camera.set('type', 'perspective')
        if camera.sensor_fit == 'AUTO':
            if bpy.context.scene.render.resolution_x < bpy.context.scene.render.resolution_y:
                xml_property.set("fovY", str(degrees(camera.angle_x)))
            else:
                fovX = camera.angle_x
                scale = bpy.context.scene.render.resolution_y / bpy.context.scene.render.resolution_x
                fovY = atan(tan(fovX / 2.0) * scale) * 2.0
                xml_property.set("fovY", str(degrees(fovY)))
        elif camera.sensor_fit == 'HORIZONTAL':
            fovX = camera.angle_x
            scale = bpy.context.scene.render.resolution_y / bpy.context.scene.render.resolution_x
            fovY = atan(tan(fovX / 2.0) * scale) * 2.0
            xml_property.set("fovY", str(degrees(fovY)))
        elif camera.sensor_fit == 'VERTICAL':
            xml_property.set("fovY", str(degrees(camera.angle_y)))
        xml_property.set("nearZ", str(camera.clip_start))
        xml_property.set("farZ", str(camera.clip_end))
    elif camera.type == 'ORTHO':
        xml_camera.set('type', 'orthographic')
    xml_camera.set('width', str(bpy.context.scene.render.resolution_x))
    xml_camera.set('height', str(bpy.context.scene.render.resolution_y))
    write_xml_transform(xml_camera, object)

def write_xml_light(xml_scene, object):
    light = object.data
    xml_light = ET.SubElement(xml_scene, 'light')
    xml_property = ET.SubElement(xml_light, 'property')
    if light.type == 'POINT':
        radius = light.shadow_soft_size
        xml_light.set('type', 'point')
        xml_property.set('radius', str(radius))
        # power w -> radiance
        area = 4.0 * pi * radius * radius
        radiance = (light.color * light.energy) / (pi * area)
        xml_property.set('radiance', f'{radiance[0]} {radiance[1]} {radiance[2]}')
    elif light.type == 'SUN':
        xml_light.set('type', 'directional')
        xml_property.set('angularDiameter', str(degrees(light.angle)))
        # irradiance w/(m^2) -> radiance
        cosAngle = cos(light.angle * 0.5)
        solidAngle = 2.0 * pi * (1.0 - cosAngle)
        radiance = (light.color * light.energy) / solidAngle
        xml_property.set('radiance', f'{radiance[0]} {radiance[1]} {radiance[2]}')
    elif light.type == 'SPOT':
        radius = light.shadow_soft_size
        xml_light.set('type', 'spot')
        xml_property.set('radius', str(radius))
        xml_property.set('outerConeAngle', str(degrees(light.spot_size)))
        xml_property.set('innerConeAngle', str(degrees(light.spot_size * (1.0 - light.spot_blend))))
        # power w -> radiance
        area = 4.0 * pi * radius * radius
        radiance = (light.color * light.energy) / (pi * area)
        xml_property.set('radiance', f'{radiance[0]} {radiance[1]} {radiance[2]}')
    elif light.type == 'AREA':
        area = 1.0
        if light.shape == 'SQUARE':
            xml_light.set('type', 'rect')
            xml_property.set('width', str(light.size))
            xml_property.set('height', str(light.size))
            area = light.size * light.size
        elif light.shape == 'RECTANGLE':
            xml_light.set('type', 'rect')
            xml_property.set('width', str(light.size))
            xml_property.set('height', str(light.size_y))
            area = light.size * light.size_y
        elif light.shape == 'DISK':
            radius = light.size * 0.5
            xml_light.set('type', 'disk')
            xml_property.set('radius', str(radius))
            area = pi * radius * radius
        # power w -> radiance
        radiance = (light.color * light.energy) / (pi * area)
        xml_property.set('radiance', f'{radiance[0]} {radiance[1]} {radiance[2]}')
    write_xml_transform(xml_light, object)

def export_xml(out_dir):
    xml_scene = ET.Element('scene')
    
    for object in bpy.context.view_layer.objects:
        if object.type == 'CAMERA':
            write_xml_camera(xml_scene, object)

    for object in bpy.context.view_layer.objects:
        if object.type == 'LIGHT':
            write_xml_light(xml_scene, object)
    
    write_xml_materials(xml_scene)

    for object in bpy.context.view_layer.objects:
        if object.type in {'MESH', 'CURVE', 'FONT', 'SURFACE'}:
            xml_mesh = ET.SubElement(xml_scene, 'mesh')
            xml_mesh.set('filepath', get_mesh_ply_path(object))
            b_mesh = object.data
            if len(b_mesh.materials) > 0:
                material = b_mesh.materials[0]
                xml_mesh.set('material', material.name_full)
    
    ET.indent(xml_scene)
    ET.ElementTree(xml_scene).write(out_dir + "scene.xml")

if __name__=='__main__':
    # preprocess_mesh()

    out_dir = 'E:/blender_export/'
    export_mesh(out_dir)
    export_xml(out_dir)