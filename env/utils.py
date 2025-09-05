from stl import mesh
import os
import mujoco as mj
import xml.etree.ElementTree as ET
import param
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

if __name__ == "__main__":
    '''
    convert_stl_from_ascii_to_binary(r"G:\irregularBPP\dataset\objaversestl\nametag_20191103-68-bhpt8a.stl",
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
    print(object_specilized_scale(r"G:\irregularBPP\dataset\objaversestl\check_rest.stl"))
    #print(object_specilized_scale(r"G:\irregularBPP\dataset\objaversestl\check_rest.stl"))
    #print(object_specilized_scale(r"G:\irregularBPP\env\franka_emika_panda\panda.xml"))
    
    