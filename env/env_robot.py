import xml.etree.ElementTree as ET
import mujoco as mj
import mujoco.viewer as viewer
import numpy as np
import datetime
import os
import sys
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__)))
import utils
from param import param
from env_init import MuJoCoEnvironmentInitializer, generate_initial_xml
from simulator import simulator
from state import N_OBJ_State

class MujocoPackingEnv:

    def __init__(self, xml_path, initial_packing_object):
        
        self.initial_xml_path = xml_path
        self.current_xml_path = xml_path
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.state_agent = N_OBJ_State(self.current_xml_path)

        self.initial_packing_object = initial_packing_object
        self.curr_packing_object = []
        self.current_item_idx = 0

    def add_object_to_scene(self, x, y, z, obj_path=None):

        if self.current_item_idx >= param.data_num:
            return

        tree = ET.parse(self.current_xml_path)
        root = tree.getroot()
        assets = root.find('asset')

        #首先设置mesh
        mesh_name = f"obj_{self.current_item_idx}"
        mesh_elem = ET.SubElement(assets, 'mesh')
        mesh_elem.set('name', mesh_name)
        mesh_elem.set('file', obj_path)

        #scale = utils.object_specilized_scale(obj_path)
        scale = param.scale
        mesh_elem.set('scale', f"{scale} {scale} {scale}")

        worldbody = root.find('worldbody')

        body_elem = ET.SubElement(worldbody, "body")
        body_elem.set("name", mesh_name)
        body_elem.set("pos", f"{x} {y} {z}")
        body_elem.set("quat", "1 0 0 0")

        # 添加自由关节 - 正确的方式
        joint_elem = ET.SubElement(body_elem, 'joint')
        joint_elem.set('name', f"freejoint_{self.current_item_idx}")
        joint_elem.set('type', 'free')

        # 添加惯性属性（质量） - 必须的！
        inertial_elem = ET.SubElement(body_elem, 'inertial')
        inertial_elem.set('pos', '0 0 0')
        inertial_elem.set('mass', '1.0')  # 设置质量
        inertial_elem.set('diaginertia', '0.01 0.01 0.01')  # 设置惯性矩

        # 添加几何体和碰撞
        geom_elem = ET.SubElement(body_elem, 'geom')
        geom_elem.set('name', f"obj_geom_{self.current_item_idx}")
        geom_elem.set('type', 'mesh')
        geom_elem.set('mesh', mesh_name)
        geom_elem.set('rgba', '0.8 0.6 0.2 1')  # 颜色
        
        # 物理属性
        geom_elem.set('friction', '0.7 0.01 0.01')  # 摩擦

        
        # 保存临时XML并重新加载模型
        temp_xml = f"scene_{self.current_item_idx}.xml"
        tree.write(os.path.join(param.result_path_now, temp_xml))

        self.model = mj.MjModel.from_xml_path(os.path.join(param.result_path_now, temp_xml))
        self.data = mj.MjData(self.model)

        
        self.current_xml_path = os.path.join(param.result_path_now, temp_xml)
        return os.path.join(param.result_path_now, temp_xml)

    def add_object_to_packing_region(self):
        
        tree = ET.parse(param.conveyor_xml)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        for geom in worldbody.findall("geom"):
            if geom.get("name") == "warehouse_floor":
                poses = geom.get("pos").split(" ")
                pos_x, pos_y, pos_z = poses[0], poses[1], poses[2]
