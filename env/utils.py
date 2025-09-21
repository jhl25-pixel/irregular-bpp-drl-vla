from stl import mesh
import os
import mujoco as mj
import xml.etree.ElementTree as ET
import param
import numpy as np
from stl import mesh

def model_scale(xml_path, write_path, scale=2):
    '''
    --> mainly scale the robot, but scale every unit. May lead to many problems
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assets = root.find("asset")
    for mesh in assets.findall("mesh"):
        mesh.set("scale", f"{scale} {scale} {scale}")
    
    tree.write(write_path)



def is_binary_stl(filename):
    """
    Check if an STL file is in binary format
    """
    try:
        # Try to read the file as a binary STL
        m = mesh.Mesh.from_file(filename)
        return True
    except Exception as e:
        print(e)
        return False

def convert_stl_from_ascii_to_binary(ascii_file_path, binary_file_path):
    """
    Robust conversion of ASCII STL to binary STL
    """
    ascii_mesh = mesh.Mesh.from_file(ascii_file_path)
        
    # 保存为二进制格式
    ascii_mesh.save(binary_file_path)
        
    #print(f"转换成功！文件已保存为: {binary_file_path}")
def judge_abnormal_loc(x, y, z):

    return abs(x) > 100 or abs(y) > 100 or abs(z) > 100

def remove_outlier(xml_path):
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    cnt = 0
    for body in worldbody.findall("body"):
        body_name = body.get("name")
        if body_name.startswith("obj_"):
            pos = [float(idx) for idx in  body.get("pos").split(" ")]
            if judge_abnormal_loc(pos[0], pos[1], pos[2]):
                worldbody.remove(body)
                cnt += 1
    tree.write(os.path.join(param.param.absolute_irbpp_root_path, "tmp", "tmp.xml"))
    path = os.path.join(param.param.absolute_irbpp_root_path, "tmp", "tmp.xml")
    print(f"{cnt} 个物体被忽略！")
    return path, cnt


#def remove_the_object_that_cannot_be_loaded():
    1
def batch_conversion():
    path = r"G:\irregularBPP\env\conveyor_system\meshes"
    for file in os.listdir(path):
        if file.endswith(".stl"):
            print(os.path.join(path, file))
            convert_stl_from_ascii_to_binary(os.path.join(path, file),
                                     os.path.join(path, file))


def object_specilized_scale(obj_path : str):
    '''
    -> analysis the scale of the object
    '''
    file = mesh.Mesh.from_file(obj_path)
    volume, cog, inertia = file.get_mass_properties()
    return param.param.obj_scale_target_volume / volume

def return_conveyor_range_worldbody(xml_path):

    '''
    return x min, y min, zmin, x max, y max, z max
    '''

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    data = []
    for bodies in worldbody.findall("geom"):
        pos = [float(i) for i in bodies.get("pos").split(" ")]
        if bodies.get("name")=="belt_surface":
            sizes = [float(i) for i in bodies.get("size").split(" ")]
            data = [pos[i] - sizes[i] for i in range(len(pos))], [pos[i] + sizes[i] for i in range(len(pos))]
    return np.array(data)

def return_connector_range_worldbody(xml_path):

    '''
    return x min, y min, zmin, x max, y max, z max
    '''

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    data = []
    for bodies in worldbody.findall("geom"):
        pos = [float(i) for i in bodies.get("pos").split(" ")]
        if bodies.get("name")=="connection_bridge":
            sizes = [float(i) for i in bodies.get("size").split(" ")]
            data = [pos[i] - sizes[i] for i in range(len(pos))], [pos[i] + sizes[i] for i in range(len(pos))]
    return np.array(data)

def return_host_range_worldbody(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    data = []
    for bodies in worldbody.findall("geom"):
        pos = [float(i) for i in bodies.get("pos").split(" ")]
        if bodies.get("name")=="warehouse_floor":
            sizes = [float(i) for i in bodies.get("size").split(" ")]
            data = [pos[i] - sizes[i] for i in range(len(pos))], [pos[i] + sizes[i] for i in range(len(pos))]
    return np.array(data)


def return_collection_box_range_worldbody(xml_path):
    """
    Compute an axis-aligned bounding box (min, max) in world coordinates for the
    body named 'collection_box' by aggregating its child geoms (box types use
    the 'size' attribute as half-sizes). Returns (min, max) as two numpy arrays
    with shape (3,). If the body or sizes cannot be found, returns a reasonable
    default around the body's pos.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # find collection_box body
    for body in worldbody.findall('body'):
        if body.get('name') == 'collection_box':
            # body world position
            body_pos = np.array([float(x) for x in body.get('pos', '0 0 0').split()])
            mins = []
            maxs = []
            # look for geoms under this body
            for geom in body.findall('geom'):
                pos = geom.get('pos', '0 0 0')
                pos = np.array([float(x) for x in pos.split()]) + body_pos
                size_attr = geom.get('size')
                if size_attr:
                    half = np.array([float(x) for x in size_attr.split()])
                    mins.append(pos - half)
                    maxs.append(pos + half)
                else:
                    # if no size (e.g. mesh), try to use small epsilon around pos
                    eps = np.array([0.05, 0.05, 0.05])
                    mins.append(pos - eps)
                    maxs.append(pos + eps)

            if len(mins) > 0:
                overall_min = np.min(np.stack(mins, axis=0), axis=0)
                overall_max = np.max(np.stack(maxs, axis=0), axis=0)
                return np.array(overall_min), np.array(overall_max)
            else:
                # fallback: return body_pos +/- defaults
                default_min = body_pos + np.array([-1.4, -1.4, -0.4])
                default_max = body_pos + np.array([1.4, 1.4, 0.9])
                return np.array(default_min), np.array(default_max)

    # if not found, fallback to origin-centered default
    return np.array([-1.4, -1.4, -0.4]), np.array([1.4, 1.4, 0.9])


def is_on_box(body_pos, box_min, box_max, tolerance: float = 0.0):
    """
    Return True if body_pos (iterable of length 3) lies within the axis-aligned
    box defined by box_min and box_max (both array-like length 3). tolerance is
    applied as an extra margin (in world units) around the box.
    """
    p = np.array(body_pos)
    min_arr = np.array(box_min) - tolerance
    max_arr = np.array(box_max) + tolerance
    return bool(np.all(p >= min_arr) and np.all(p <= max_arr))


def is_on_conveyor(body_pos, conveyor_min, conveyor_max, tolerance=0.1):
    """
    检查物体是否在传送带表面上
    tolerance: Z轴方向的容差，用于判断是否在表面上
    """
    x, y, z = body_pos
    x_min, y_min, z_min = conveyor_min
    x_max, y_max, z_max = conveyor_max
    
    # 检查XY平面是否在传送带范围内，Z轴是否在表面附近
    return (x_min <= x <= x_max and 
            y_min <= y <= y_max and 
            abs(z - z_max) <= tolerance)

def is_on_host(body_pos, host_min, host_max, tolerance=0.1):

    x, y, z = body_pos
    x_min, y_min, z_min = host_min
    x_max, y_max, z_max = host_max
    
    # 检查XY平面是否在传送带范围内，Z轴是否在表面附近
    return (x_min <= x <= x_max and 
            y_min <= y <= y_max and 
            abs(z - z_max) <= tolerance)

def is_on_bridge(body_pos, bridge_min, bridge_max, tolerance=0.1):

    x, y, z = body_pos
    x_min, y_min, z_min = bridge_min
    x_max, y_max, z_max = bridge_max
    
    # 检查XY平面是否在传送带范围内，Z轴是否在表面附近
    return (x_min <= x <= x_max and 
            y_min <= y <= y_max and 
            abs(z - z_max) <= tolerance)

def compute_xyaxes_for_certain_direction(location_target : list, location_camera : list):
    import numpy as np
    assert len(location_target) == len(location_camera)
    z_vector = np.array([location_target[i] - location_camera[i] for i in range(len(location_target))])
    if abs(z_vector[1]) < 0.001:
        y_vector = np.array([0, 1.0, 0])
    else:
        y_vector = np.array([1.0, -z_vector[0] / z_vector[1], 0])

    x_vector = np.cross(z_vector, y_vector)
    if np.dot(np.cross(z_vector, y_vector), z_vector) < 0:
        y_vector = -y_vector
    return x_vector, y_vector

def return_camera_position(xml_path, camera_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    cameras = worldbody.findall("camera")
    for camera in cameras:
        if camera_name == camera.get("name"):
            return [float(i) for i in camera.get("pos").split(" ")]
    

def set_camera_direction(xml_path, camera_name, xyaxes):
    '''
    note that xyaxes refers to "a b c d e f", the axes
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    cameras = worldbody.findall("camera")
    for camera in cameras:
        if camera_name == camera.get("name"):
            camera.set("xyaxes", " ".join([str(i) for i in xyaxes]))

    tree.write(xml_path)

def return_image(camera_name='wrist_cam', width=640, height=480):
    '''
    伪代码，视你的 mujoco 版本而定
    '''
    import renderer
    img = renderer.render(camera=camera_name, width=640, height=480)
    return img

if __name__ == "__main__":
    #import numpy as np
    #print(return_camera_position(param.param.conveyor_xml, "wrist_cam"))
    #x_v, y_v = compute_xyaxes_for_certain_direction([0, 0, 0], return_camera_position(param.param.conveyor_xml, "wrist_cam"))
    #print(np.concatenate((x_v, y_v)))
    #set_camera_direction(param.param.conveyor_xml, "wrist_cam", np.concatenate((x_v, y_v)))
    #print(x)
    '''
    convert_stl_from_ascii_to_binary(r"G:\irregularBPP\dataset\objav ersestl\nametag_20191103-68-bhpt8a.stl",
                                     r"G:\irregularBPP\dataset\objaversestl\nametag_20191103-68-bhpt8a2.stl")
    print(is_binary_stl(r"G:\irregularBPP\dataset\objaversestl\nametag_20191103-68-bhpt8a.stl"))
    print(is_binary_stl(r"G:\irregularBPP\dataset\objaversestl\nametag_20191103-68-bhpt8a2.stl"))
    '''
    #_, cnt = remove_outlier(r"G:\irregularBPP\experiment\0\finish_drop_scene_8.xml")
    #print(cnt)
    '''
    model_scale(r"G:\irregularBPP\env\franka_emika_panda\panda.xml",
                r"G:\irregularBPP\env\franka_emika_panda\panda2.xml",
                scale=2)
    '''
    #print(object_specilized_scale(r"G:\irregularBPP\dataset\objaversestl\single_joycon_grip-.STL"))
    #print(object_specilized_scale(r"G:\irregularBPP\dataset\objaversestl\stockpart.stl"))
    #print(object_specilized_scale(r"G:\irregularBPP\env\franka_emika_panda\panda.xml"))
    #print(return_conveyor_range_worldbody(r"G:\irregularBPP\env\franka_emika_panda\conveyor.xml"))
    #print(return_collection_box_range_worldbody(r"D:\lab\irregular-bpp-drl-vla\env\franka_emika_panda\conveyor_system.xml"))
    #min_range, max_range = return_collection_box_range_worldbody(r"D:\lab\irregular-bpp-drl-vla\env\franka_emika_panda\conveyor_system.xml")
    #print(is_on_box([0, 0, 0], min_range, max_range, tolerance=0.0))

    
    