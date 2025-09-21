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
'''
usage : positions, reward, done, _ = env.step(test_loc)

here, the "reward" means the reward obtained from one single step



'''

class rewarder:

    def __init__(self, xml_path, positions, object_path):
        self.xml_path = xml_path
        self.positions = positions
        self.object_path = object_path
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.reward = 0

    def return_util_reward(self, object_name):
        """
        return the reward in the single step
        """

        try:
            # 解析XML
            if os.path.exists(self.xml_path):
                tree = ET.parse(self.xml_path)
                root = tree.getroot()
            else:
                root = ET.fromstring(self.xml_path)
        except:
            return 0.0

        # 获取盒子边界（简化的内部空间估计）
        collection_box = root.find(".//body[@name='collection_box']")
        if collection_box is None:
            return 0.0
        
        parent_pos = [float(coord) for coord in collection_box.get('pos', '0 0 0').split()]
        
        # 简化的内部空间边界估计
        inner_x_min, inner_x_max = -1.4, 1.4    # 考虑墙壁厚度
        inner_y_min, inner_y_max = -1.4, 1.4    # 考虑墙壁厚度  
        inner_z_min, inner_z_max = -0.4, 0.9    # 从底部上表面到墙壁顶部
        
        # 转换到世界坐标系
        inner_x_min += parent_pos[0]
        inner_x_max += parent_pos[0]
        inner_y_min += parent_pos[1] 
        inner_y_max += parent_pos[1]
        inner_z_min += parent_pos[2]
        inner_z_max += parent_pos[2]
        
        # 检查所有物体
        total_reward = 0
        obj_count = 0
        
        obj_data = self.positions[object_name]
        obj_pos = obj_data['pos']

        if (inner_x_min <= obj_pos[0] <= inner_x_max and
            inner_y_min <= obj_pos[1] <= inner_y_max and
            inner_z_min <= obj_pos[2] <= inner_z_max):

            total_reward += 1
        
        return total_reward



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
    


    def simulate_drop(self, max_steps=2000):

        mj.mj_resetData(self.model, self.data)

        stable_threshold = param.stable_threshold_v
        stable_count = 0
        required_stable_steps = param.required_stable_steps


        for step in range(max_steps):

            mj.mj_step(self.model, self.data)

            total_velocity = 0

            for i in range(self.model.nbody):
                if self.model.body(i).name.startswith('obj_'):
                    vel = np.linalg.norm(self.data.body(i).cvel)
                    total_velocity += vel

            if total_velocity < stable_threshold:
                stable_count += 1
            else:
                stable_count = 0
            
            if stable_count >= required_stable_steps:
                break

        positions = self.get_final_positions()

        return positions
    
    def get_final_positions(self):

        positions = {}
        for i in range(self.model.nbody):
            body_name = self.model.body(i).name
            if body_name.startswith("obj_"):
                pos = self.data.body(i).xpos.copy()
                quat = self.data.body(i).xquat.copy()
                positions[body_name] = {'pos' : pos, 'quat' : quat}

        
        return positions

    def set_final_positions_in_xml(self, positions):

        
        tree = ET.parse(self.current_xml_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')

        for body in worldbody.findall("body"):
            body_name = body.get("name")
            if body_name.startswith("obj_"):
                pos, quat = positions[body_name]['pos'], positions[body_name]['quat']
                pos_format = f"{pos[0]} {pos[1]} {pos[2]}"
                quat_format = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
                body.set("pos", pos_format)
                body.set("quat", quat_format)

        temp_xml = f"finish_drop_scene_{self.current_item_idx}.xml"
        tree.write(os.path.join(param.result_path_now, temp_xml))

        self.model = mj.MjModel.from_xml_path(os.path.join(param.result_path_now, temp_xml))
        self.data = mj.MjData(self.model)
        
        self.current_xml_path = os.path.join(param.result_path_now, temp_xml)
    

    def step(self, initial_positions, packing_object=None):
        '''
        input:
        initial_positions : the initial throw position before drop simulation

        output:
        positions : The cumulated position list, in the form of dictionary
        '''
        if packing_object == None:
            packing_object = self.initial_packing_object[self.current_item_idx]

        self.add_object_to_scene(initial_positions[0], initial_positions[1], initial_positions[2], packing_object)

        positions = self.simulate_drop()
        print(positions)
        self.set_final_positions_in_xml(positions)
        
        self.state_agent = N_OBJ_State(self.current_xml_path)
        state = self.state_agent.return_the_state()

        object_name = f"obj_{self.current_item_idx}"
        reward_model = rewarder(self.current_xml_path, positions, self.initial_packing_object[self.current_item_idx])
        reward = reward_model.return_util_reward(object_name)

        done = self.current_item_idx >= len(self.initial_packing_object) - 1
        self.current_item_idx += 1

        return state, reward, done, {}






def build_the_env():
    xml_file = generate_initial_xml(
        box_scale=0.2, 
        box_position=(6, 2, 0.1),
        conveyor_length=2.5,
        conveyor_width=0.6,
        conveyor_position=(-10, 10, 0.5),
        object_num=25
    )

    data_simulator = simulator(param.data_path, data_type="stl")
    simulated_object_list = data_simulator._roll_the_dice(param.data_num)

    env = MujocoPackingEnv(xml_file, simulated_object_list)
    env.init()
    return env

def test_one_step():
    total_reward = 0
    total_count = 0
    env = build_the_env()
    for epoch in range(param.epoches):
        test_loc = [0, 0, 0]
        state, reward, done, _ = env.step(test_loc)
        total_reward += reward
        total_count += 1
        print(f"current state is {state}")
        print(f"目前为第{epoch + 1}个 epoch")
        print(f"current reward is {reward}")
        if done: 
            break



def test_convor_location():
    env.step([-1.5, -0.4, 1.5], r"G:\irregularBPP\dataset\objaversestl\SurfaceMountV3Right.stl")
    env.step([-2, -0.4, 1.5], r"G:\irregularBPP\dataset\objaversestl\stockpart.stl")
    env.step([1, -0.4, 1.5], r"G:\irregularBPP\dataset\objaversestl\single_joycon_grip-.STL")
    env.step([1.5, -0.4, 1.5], r"G:\irregularBPP\dataset\objaversestl\ShroomTopRemix.stl")

def test_hard_structure_of_robotics():
    env.step([-0.7, 0, 5.5], r"G:\irregularBPP\dataset\objaversestl\SurfaceMountV3Right.stl")

def test_useful_objects():
    #env.step([-1.3, -0.4, 0.6], r"G:\irregularBPP\dataset\objaversestl\SurfaceMountV3Right.stl") #L形状物体
    env.step([-1.2, -0.4, 0.6], r"G:\irregularBPP\dataset\objaversestl\stockpart.stl") #船形物体
    env.step([1, -0.4, 0.6], r"G:\irregularBPP\dataset\objaversestl\single_joycon_grip-.STL") 
    #env.step([1.2, -0.4, 0.3], r"G:\irregularBPP\dataset\objaversestl\ShroomTopRemix.stl")

if __name__ == "__main__":
    
    #test_one_step()
    env = build_the_env()
    #env.step([-1.3, -0.4, 0.6], r"G:\irregularBPP\dataset\objaversestl\SurfaceMountV3Right.stl") #L形状物体
    env.step([-4.2, -0.5, 0.3], r"G:\irregularBPP\dataset\objaversestl\stockpart.stl") #船形物体
    env.step([-3.7, -0.5, 0.3], r"G:\irregularBPP\dataset\objaversestl\single_joycon_grip-.STL")
    env.step([-3.4, -0.5, 0.3], r"G:\irregularBPP\dataset\objaversestl\stockpart.stl")

    #env.step([1.2, -0.4, 0.3], r"G:\irregularBPP\dataset\objaversestl\ShroomTopRemix.stl")