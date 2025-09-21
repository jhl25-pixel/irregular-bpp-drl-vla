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
import conveyor
import torch

class MujocoPackingEnv:

    def __init__(self, xml_path, initial_packing_object):
        
        self.initial_xml_path = xml_path
        self.current_xml_path = xml_path
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer()
        self.renderer.set_model(self.model)
        self.state_agent = N_OBJ_State(self.current_xml_path)

        self.initial_packing_object = initial_packing_object
        self.curr_packing_object = []
        self.current_item_idx = 0
        # renderer will be created on demand (recreated when model changes)
        self.renderer = None
        # camera names defined in conveyor.xml
        # front_cam: front view, side_cam: side view, topdown_cam: top-down, egocentric_cam: robot-eye, wrist_cam: wrist
        self.camera_names = ["front_cam", "side_cam", "topdown_cam", "egocentric_cam", "wrist_cam"]

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
        # model changed -> drop renderer so it will be recreated lazily
        self.renderer = None

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
    
    def return_image(self, camera_name='wrist_cam', width=640, height=480):
        '''
        Return an RGB image from the specified camera.

        Args:
            camera_name (str): Name of the camera as defined in the XML.
            width (int): Width of the output image.
            height (int): Height of the output image.
        '''
        img = self.renderer.render(camera=camera_name, width=width, height=height)
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
        # prepare action vector and write to ctrl
        if action is None:
            action_arr = np.zeros(self.model.nu, dtype=np.float32)
        else:
            action_arr = np.array(action, dtype=np.float32).ravel()

        # ensure ctrl has correct length
        try:
            self.data.ctrl[:] = 0.0
            n = min(self.model.nu, action_arr.size)
            if n > 0:
                self.data.ctrl[:n] = action_arr[:n]
        except Exception:
            # some models might not have actuators
            pass

        # step simulation
        for _ in range(n_substeps):
            # apply conveyor simple velocity to objects on belt/host/bridge
            try:
                conveyor.apply_conveyor_velocity_simple(self.model, self.data, param.conveyor_xml, conveyor_speed=param.conveyor_speed)
            except Exception:
                # don't fail if conveyor utilities aren't available
                pass

            mj.mj_step(self.model, self.data)

        # collect object states from current simulation data
        positions = {}
        obj_states = []
        for i in range(self.model.nbody):
            body_name = self.model.body(i).name
            if body_name and body_name.startswith("obj_"):
                pos = self.data.body(i).xpos.copy()
                quat = self.data.body(i).xquat.copy()
                positions[body_name] = {'pos': pos, 'quat': quat}
                obj_states.append(np.concatenate((pos, quat)).astype(np.float32))

        # pad to fixed length (param.obj_num)
        if len(obj_states) < param.obj_num:
            for _ in range(param.obj_num - len(obj_states)):
                obj_states.append(np.zeros(7, dtype=np.float32))

        if len(obj_states) > 0:
            obs = np.concatenate(obj_states, axis=0)
        else:
            obs = np.zeros(param.obj_num * 7, dtype=np.float32)

        state = torch.FloatTensor(obs)

        # simple reward: check whether the most recently added object is inside collection_box
        reward = 0.0
        last_idx = max(0, self.current_item_idx - 1)
        last_name = f"obj_{last_idx}"
        if last_name in positions:
            # load collection_box position from current xml (fallback to origin)
            try:
                tree = ET.parse(self.current_xml_path)
                root = tree.getroot()
                # use utility to compute collection box world-aligned bounds
                try:
                    box_min, box_max = utils.return_collection_box_range_worldbody(self.current_xml_path)
                except Exception:
                    # fallback to hardcoded defaults if util fails
                    box_min = np.array([-1.4, -1.4, -0.4])
                    box_max = np.array([1.4, 1.4, 0.9])

                p = positions[last_name]['pos']
                try:
                    if utils.is_on_box(p, box_min, box_max, tolerance=0.0):
                        reward = 1.0
                except Exception:
                    # final fallback: simple axis-aligned check
                    if (box_min[0] <= p[0] <= box_max[0] and
                        box_min[1] <= p[1] <= box_max[1] and
                        box_min[2] <= p[2] <= box_max[2]):
                        reward = 1.0
            except Exception:
                reward = 0.0

        # advance item counter if desired (user may want manual control); here we do not auto-increment
        done = (self.current_item_idx >= len(self.initial_packing_object) - 1)
        info = {'positions': positions}

        # optional: render cameras and include images in info
        if render:
            images = {}
            try:
                if self.renderer is None:
                    try:
                        # preferred constructor
                        self.renderer = mj.Renderer(self.model)
                    except Exception:
                        try:
                            # fallback: create empty renderer and set model
                            self.renderer = mj.Renderer()
                            if hasattr(self.renderer, 'set_model'):
                                self.renderer.set_model(self.model)
                        except Exception:
                            self.renderer = None

                if self.renderer is not None:
                    for cam in self.camera_names:
                        try:
                            # try update_scene with data first
                            try:
                                self.renderer.update_scene(self.data, camera=cam)
                            except Exception:
                                try:
                                    self.renderer.update_scene(self.model, self.data, camera=cam)
                                except Exception:
                                    pass

                            # try render
                            try:
                                img = self.renderer.render()
                            except Exception:
                                try:
                                    img = self.renderer.render(camera=cam)
                                except Exception:
                                    img = None

                            images[cam] = img
                        except Exception:
                            images[cam] = None
            except Exception:
                images = {}

            info['images'] = images

        return state, float(reward), bool(done), info


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

if __name__ == "__main__":
    env = build_the_env()
    print(env.return_image('wrist_cam', 320, 240).shape)