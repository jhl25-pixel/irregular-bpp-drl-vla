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
from viewer import save_image_to_video, Images
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

param.res_idx = timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class MujocoPackingEnv:

    def __init__(
            self, xml_path, initial_packing_object, robot="pandas", vla="pi05_droid", width=640, height=480
        ):
        system = platform.system().lower()
        if system == 'linux':
            os.environ['MUJOCO_GL'] = 'osmesa'

        print(xml_path,"asddsad")
        self.initial_xml_path = xml_path
        self.current_xml_path = xml_path
        self.width = width
        self.height = height
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        if robot == "pandas":
            self.robot = model.Panda(self.model, self.data)
        self.p0 = model.Pi0(VLA_INFO[vla]['config_name'], VLA_INFO[vla]['checkpoint_path'])
        self.renderer = mj.Renderer(self.model, width=self.width, height=self.height)
        self.state_agent = N_OBJ_State(self.current_xml_path)

        self.initial_packing_object = initial_packing_object
        self.curr_packing_object = []
        self.current_item_idx = 0
        # camera names defined in conveyor.xml
        # front_cam: front view, side_cam: side view, topdown_cam: top-down, egocentric_cam: robot-eye, wrist_cam: wrist
        self.camera_names = ["front_cam", "side_cam", "topdown_cam", "egocentric_cam", "wrist_cam"]

        self.action_buffer = None
        self.action_step_idx = 0
        self.step_count = 0
        self.images = Images()

    def init(self):
        result_path = os.path.join(param.result_path, str(param.res_idx))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        param.result_path_now = result_path
        assets_destination = os.path.join(param.result_path_now, 'assets')
        if not os.path.exists(assets_destination):
            shutil.copytree(param.robot_assets, assets_destination)
        
        shutil.copy(param.conveyor_xml, param.result_path_now)
        shutil.copy(param.robot_xml_full_location, param.result_path_now)

    def add_object_to_scene(self, x, y, z, obj_path=None):
        if self.current_item_idx >= param.data_num:
            return

        # 保存当前状态
        old_qpos = self.data.qpos.copy()
        old_qvel = self.data.qvel.copy() 
        old_ctrl = self.data.ctrl.copy()
        
        print(f"添加物体到位置: ({x}, {y}, {z})")

        tree = ET.parse(self.current_xml_path)
        root = tree.getroot()
        assets = root.find('asset')

        # 设置mesh
        mesh_name = f"obj_{self.current_item_idx}"
        mesh_elem = ET.SubElement(assets, 'mesh')
        mesh_elem.set('name', mesh_name)
        mesh_elem.set('file', obj_path)

        scale = param.scale
        mesh_elem.set('scale', f"{scale} {scale} {scale}")

        worldbody = root.find('worldbody')

        body_elem = ET.SubElement(worldbody, "body")
        body_elem.set("name", mesh_name)
        body_elem.set("pos", f"{x} {y} {z}")  # XML中的位置
        body_elem.set("quat", "1 0 0 0")

        # 添加自由关节
        joint_elem = ET.SubElement(body_elem, 'joint')
        joint_elem.set('name', f"freejoint_{self.current_item_idx}")
        joint_elem.set('type', 'free')

        # 添加惯性属性
        inertial_elem = ET.SubElement(body_elem, 'inertial')
        inertial_elem.set('pos', '0 0 0')
        inertial_elem.set('mass', '1.0')
        inertial_elem.set('diaginertia', '0.01 0.01 0.01')

        # 添加几何体和碰撞
        geom_elem = ET.SubElement(body_elem, 'geom')
        geom_elem.set('name', f"obj_geom_{self.current_item_idx}")
        geom_elem.set('type', 'mesh')
        geom_elem.set('mesh', mesh_name)
        
        geom_elem.set('rgba', '0.8 0.6 0.2 1')
        geom_elem.set('friction', '0.7 0.01 0.01')

        for body in worldbody.findall("body"):
            name = body.get("name")
            if name and name.startswith("obj_"):
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    jntadr = self.model.body_jntadr[body_id]
                    if jntadr >= 0:
                        qpos_adr = self.model.jnt_qposadr[jntadr]
                        # 取当前qpos中的位置和四元数
                        x = self.data.qpos[qpos_adr + 0]
                        y = self.data.qpos[qpos_adr + 1]
                        z = self.data.qpos[qpos_adr + 2]
                        qw = self.data.qpos[qpos_adr + 3]
                        qx = self.data.qpos[qpos_adr + 4]
                        qy = self.data.qpos[qpos_adr + 5]
                        qz = self.data.qpos[qpos_adr + 6]
                        body.set("pos", f"{x} {y} {z}")
                        body.set("quat", f"{qw} {qx} {qy} {qz}")

        xml_nextstage = os.path.join(param.result_path_now, f"scene_{self.current_item_idx}.xml")
        tree.write(xml_nextstage)
        
        # 重新加载模型
        old_model = self.model
        self.model = mj.MjModel.from_xml_path(xml_nextstage)
        self.data = mj.MjData(self.model)
        
        # 恢复旧物体状态
        min_qpos_len = min(len(old_qpos), len(self.data.qpos))
        min_qvel_len = min(len(old_qvel), len(self.data.qvel))
        min_ctrl_len = min(len(old_ctrl), len(self.data.ctrl))
        
        self.data.qpos[:min_qpos_len] = old_qpos[:min_qpos_len]
        self.data.qvel[:min_qvel_len] = old_qvel[:min_qvel_len] 
        self.data.ctrl[:min_ctrl_len] = old_ctrl[:min_ctrl_len]
        
        # 🔥 关键修复：手动设置新物体的正确位置
        new_body_name = f"obj_{self.current_item_idx}"
        new_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, new_body_name)
        
        if new_body_id >= 0:
            print(f"找到新物体 {new_body_name}，body_id = {new_body_id}")
            
            # 找到新物体的关节
            body_jntadr = self.model.body_jntadr[new_body_id]
            if body_jntadr >= 0:
                # 获取关节的qpos地址
                qpos_adr = self.model.jnt_qposadr[body_jntadr]
                print(f"新物体qpos地址: {qpos_adr}")
                
                # 手动设置新物体的位置和姿态
                self.data.qpos[qpos_adr + 0] = x      # X位置
                self.data.qpos[qpos_adr + 1] = y      # Y位置  
                self.data.qpos[qpos_adr + 2] = z      # Z位置
                self.data.qpos[qpos_adr + 3] = 1.0    # 四元数w
                self.data.qpos[qpos_adr + 4] = 0.0    # 四元数x
                self.data.qpos[qpos_adr + 5] = 0.0    # 四元数y
                self.data.qpos[qpos_adr + 6] = 0.0    # 四元数z
                
                print(f"设置新物体位置为: ({x}, {y}, {z})")
            else:
                print(f"警告：物体 {new_body_name} 没有找到关节")
        else:
            print(f"警告：没有找到物体 {new_body_name}")
        
        # 更新机器人引用
        if hasattr(self, 'robot'):
            self.robot.model = self.model
            self.robot.data = self.data
        
        self.renderer = mj.Renderer(self.model, width=self.width, height=self.height)
        # 更新衍生量 - 这会根据qpos计算正确的xpos
        mj.mj_forward(self.model, self.data)
        
        # 验证物体位置
        if new_body_id >= 0:
            actual_pos = self.data.xpos[new_body_id]
            print(f"物体实际位置: {actual_pos}")
        
        self.current_xml_path = xml_nextstage
        print(f"object {self.current_item_idx} has been loaded in the environment")

        return xml_nextstage



    def add_object_to_packing_region(self):
        if self.current_item_idx >= param.data_num:
            return self.current_xml_path
            
        tree = ET.parse(param.conveyor_xml)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        
        for geom in worldbody.findall("geom"):
            if geom.get("name") == "warehouse_floor":
                poses = geom.get("pos").split(" ")
                pos_x, pos_y, pos_z = float(poses[0]), float(poses[1]), float(poses[2])
                
                print(f"warehouse_floor位置: ({pos_x}, {pos_y}, {pos_z})")
                target_x, target_y, target_z = pos_x, pos_y, pos_z + 0.5
                print(f"物体目标位置: ({target_x}, {target_y}, {target_z})")
                
                self.add_object_to_scene(target_x, target_y, target_z, 
                                    obj_path=self.initial_packing_object[self.current_item_idx])
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
        self.renderer = mj.Renderer(self.model, width=self.width, height=self.height)
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
        camera_records = {}

        
        front_image = self.return_image("front_cam")
        self.images.append(front_image)
        import cv2
        cv2.imwrite(os.path.join(param.img_save_path, "ssk.png"), front_image)
        wrist_image = self.return_image("wrist_cam")
        

        if self.action_buffer is None or self.action_step_idx >= len(self.action_buffer):
            obj_states = {
                "prompt": "pick up the object and place it in the box",
                "observation/exterior_image_1_left": front_image,
                "observation/wrist_image_left": wrist_image,
                "observation/joint_position": self.robot.get_arm_joint_position(data=self.data),
                "observation/gripper_position": self.robot.get_gripper_position(data=self.data)
            }
            self.action_buffer = self.p0.generate_action(obj_states)  # (15, 8)
            self.action_step_idx = 0
            
        # 使用当前步的动作
        current_action = self.action_buffer[self.action_step_idx]  # (8,)
        action_arr = np.array(current_action, dtype=np.float32)
        action_arr = np.clip(action_arr, -3.0, 3.0)
        
        self.action_step_idx += 1

        self.data.ctrl[:n] = action_arr[:n]

        if self.step_count % 10 == 0:  # 每10步打印一次
            print(f"=== Step {self.step_count} ===")
            for body_id in range(1, self.model.nbody):
                body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)
                pos_before = self.data.xpos[body_id].copy()
                print(f"物体 {body_name} 位置: {pos_before}")
        

        # step simulation
        for _ in range(n_substeps):

            conveyor.apply_conveyor_velocity_simple(self.model, self.data, param.conveyor_xml, conveyor_speed=param.conveyor_speed)
            mj.mj_step(self.model, self.data)

        print(f"目前在跑第{self.step_count}个epoch")
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
    print(xml_file)
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
    num_step = 0

    for i in range(param.epoches):
        env.step()
    
    for i in range(len(env.images.return_img())):
        plt.imsave(os.path.join(param.img_save_path, f"{i:03d}.png"), env.images.return_img()[i])
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_image_to_video(env.images.return_img(), output_path=f"/home/hljin/irregular-bpp-drl-vla/video/simulation_{timestamp}.mp4")