import xml.etree.ElementTree as ET
import mujoco as mj
import mujoco.viewer as viewer
import numpy as np
import datetime
import os
os.environ['MUJOCO_GL'] = 'osmesa'
import sys
import shutil
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vla', 'openpi', 'src'))
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

import utils
from param import param
from env_init import MuJoCoEnvironmentInitializer, generate_initial_xml
from simulator import simulator
from state import N_OBJ_State
import conveyor
import torch
import platform

import model
ROBOT_LIST=['pandas']
ROBOT_INFO={
    'pandas':{
        'model_path': param.robot_xml_full_location,
        'gripper_joint_names': ['panda_finger_joint1', 'panda_finger_joint2'],
        'arm_joint_name': [
            'joint1', 'joint2', 'joint3', 'joint4', 
            'joint5', 'joint6', 'joint7'
        ],
    }
}
VLA_INFO={
    'pi05_droid':{
        'config_name': 'pi05_droid',
        'checkpoint_path': 'gs://openpi-assets/checkpoints/pi05_droid'
    }
}
class MujocoPackingEnv:

    def __init__(
            self, xml_path, initial_packing_object, robot="pandas", vla="pi05_droid", width=640, height=480
        ):
        system = platform.system().lower()
        if system == 'linux':
            os.environ['MUJOCO_GL'] = 'osmesa'

        
        self.initial_xml_path = xml_path
        self.current_xml_path = xml_path
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        if robot == "pandas":
            self.robot = model.Panda(self.model, self.data)
        self.p0 = model.Pi0(VLA_INFO[vla]['config_name'], VLA_INFO[vla]['checkpoint_path'])
        self.renderer = mj.Renderer(self.model, width=width, height=height)
        self.state_agent = N_OBJ_State(self.current_xml_path)

        self.initial_packing_object = initial_packing_object
        self.curr_packing_object = []
        self.current_item_idx = 0
        # camera names defined in conveyor.xml
        # front_cam: front view, side_cam: side view, topdown_cam: top-down, egocentric_cam: robot-eye, wrist_cam: wrist
        self.camera_names = ["front_cam", "side_cam", "topdown_cam", "egocentric_cam", "wrist_cam"]
        self.step_count = 0

    def init(self):
        result_path = os.path.join(param.result_path, str(param.res_idx))
        param.res_idx += 1
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        param.result_path_now = result_path
        destination = os.path.join(param.result_path_now, 'assets')
        if not os.path.exists(destination):
            shutil.copytree(param.robot_assets, destination)

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
        if self.current_item_idx >= param.data_num:
            return self.current_xml_path
        tree = ET.parse(param.conveyor_xml)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        for geom in worldbody.findall("geom"):
            if geom.get("name") == "warehouse_floor":
                poses = geom.get("pos").split(" ")
                pos_x, pos_y, pos_z = poses[0], poses[1], poses[2]
                self.add_object_to_scene(float(pos_x), float(pos_y), float(pos_z)+0.5, obj_path=self.initial_packing_object[self.current_item_idx])
                self.curr_packing_object.append(self.initial_packing_object[self.current_item_idx])
                self.current_item_idx += 1 
                break
        return self.current_xml_path
    
    def return_image(self, camera_name='wrist_cam', width=1920, height=1440):
        '''
        Return an RGB image from the specified camera.

        Args:
            camera_name (str): Name of the camera as defined in the XML.
            width (int): Width of the output image.
            height (int): Height of the output image.
        '''
        mj.mj_forward(self.model, self.data) # 更新数据, 注意需要手动更新时间 model.time 或者使用model.mj_step()
        self.renderer.update_scene(self.data, camera_name) # 更新场景

        img = self.renderer.render() # 返回一个numpy数组
    
        return img

    def step(self, action=None, n_substeps:int=10, render:bool=False):
        """
        Apply an action to the robot actuators and step the MuJoCo simulation.

        Args:
            action: array-like, values to write into `data.ctrl`. If shorter than
                the model's actuator dimension it will fill the prefix; missing
                entries are set to 0. If None, zeros are applied.
            n_substeps: number of physics substeps (mj_step calls) to run for one env step.
            render: reserved for future use (no-op here).

        Returns:
            state (torch.FloatTensor): concatenated obj states (obj_num * 7)
            reward (float): simple reward (1 if last object inside collection_box)
            done (bool): True if we've processed all initial_packing_object
            info (dict): debug info (positions mapping)
        """
        if self.current_item_idx < param.data_num and self.step_count % param.packing_object_every_N_steps == 0:
            self.add_object_to_packing_region()
        
        # prepare action vector and write to ctrl
        if action is None:
            action_arr = np.zeros(self.model.nu, dtype=np.float32)
        else:
            action_arr = np.array(action, dtype=np.float32).ravel()

        # ensure ctrl has correct length
        n = self.model.nu
        obj_states = {}
        for camera_name in self.camera_names:
            obj_states[camera_name].append(self.return_image(camera_name))
        
        obj_states["prompt"] = "pick up the object and place it in the box"
        obj_states["observation/exterior_image_1_left"] = obj_states["front_cam"]
        obj_states["observation/joint_position"] = self.robot.get_arm_joint_positions()
        obj_states["observation/gripper_position"] = self.robot.get_gripper_position()
        action = self.p0.generate_action(obj_states)
        action_arr = np.array(action, dtype=np.float32).ravel()
        action_arr = np.clip(action_arr, -3.0, 3.0)

        self.data.ctrl[:n] = action_arr[:n]


        

        # step simulation
        for _ in range(n_substeps):

            conveyor.apply_conveyor_velocity_simple(self.model, self.data, param.conveyor_xml, conveyor_speed=param.conveyor_speed)
            mj.mj_step(self.model, self.data)

        self.step_count += 1
    

def build_the_env():
    xml_file = generate_initial_xml(
        box_scale=0.2, 
        box_position=(6, 2, 0.1),
        conveyor_length=2.5,
        conveyor_width=0.6,
        conveyor_position=(-10, 10, 0.5),
        object_num=25,
    )

    data_simulator = simulator(param.data_path, data_type="stl")
    simulated_object_list = data_simulator._roll_the_dice(param.data_num)

    env = MujocoPackingEnv(xml_file, simulated_object_list, width=640, height=480, robot="pandas", vla="pi05_droid")
    env.init()
    return env

def test():
    env = build_the_env()
    img = env.return_image('wrist_cam')
    img2 = env.return_image('topdown_cam')
    img3 = env.return_image('front_cam')
    img4 = env.return_image('side_cam')
    print(img.shape, img2.shape, img3.shape, img4.shape)
    plt.imshow(img)
    plt.savefig("test.png")
    plt.show()
    print("Image saved as test.png")
    plt.imshow(img2)
    plt.savefig("test2.png") 
    plt.show()
    print("Image saved as test2.png")
    plt.imshow(img3)
    plt.savefig("test3.png") 
    plt.show()
    print("Image saved as test3.png")
    plt.imshow(img4)
    plt.savefig("test4.png") 
    plt.show()
    print("Image saved as test4.png")

if __name__ == "__main__":
    
    env = build_the_env()
    env.step()