import torch.nn as nn
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import mujoco as mj
from abc import ABC, abstractmethod
import param
class Robot(ABC):

    def __init__(self, model, data):
        self.model = model
        self.data = data

    @abstractmethod
    def get_arm_joint_position(self):
        pass

    @abstractmethod
    def get_gripper_position(self):
        pass

    

class Panda(Robot):

    def __init__(self, model, data):
        super().__init__(model, data)
        self.arm_joint_name = [
            'joint1', 'joint2', 'joint3', 'joint4', 
            'joint5', 'joint6', 'joint7'
        ]
        self.gripper_joint_names = [
            'finger_joint1', 'finger_joint2']

        self.joint_id = []
        for arm_name in self.arm_joint_name:
            self.joint_id.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, arm_name))
        self.gripper_id = []
        for gripper_name in self.gripper_joint_names:
            self.gripper_id.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, gripper_name))

        self._reinitialize()

    def get_arm_joint_position(self, data):
        return [data.qpos[id] for id in self.joint_id]
    
    def get_gripper_position(self, data):
        return [data.qpos[id] for id in self.gripper_id]
    
    def _reinitialize(self):
        """重新初始化关节ID"""
        # 获取机械臂关节ID
        self.joint_id = []
        for arm_name in self.arm_joint_name:
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, arm_name)
            if joint_id == -1:
                print(f"Warning: 机械臂关节 {arm_name} 在模型中未找到")
            self.joint_id.append(joint_id)
        
        # 获取夹爪关节ID
        self.gripper_id = []
        for gripper_name in self.gripper_joint_names:
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, gripper_name)
            if joint_id == -1:
                print(f"Warning: 夹爪关节 {gripper_name} 在模型中未找到")
            self.gripper_id.append(joint_id)
        
        print(f"机械臂关节ID: {self.joint_id}")
        print(f"夹爪关节ID: {self.gripper_id}")

class Pi0:

    def __init__(self, config_name, checkpoint_path):
        self.config = _config.get_config(config_name)
        self.checkpoint_dir = download.maybe_download(checkpoint_path)
        self.policy = policy_config.create_trained_policy(self.config, self.checkpoint_dir)
    
    def generate_action(self, observation):
        action_chunk = self.policy.infer(observation)["actions"]
        return action_chunk

def test_load_p0():
    pi0 = Pi0("pi05_droid", "gs://openpi-assets/checkpoints/pi05_droid")
    example = {
        "observation/exterior_image_1_left": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "observation/exterior_image_2_left": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "observation/wrist_image_left": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "observation/joint_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "observation/gripper_position": [0.0],

        "prompt": "pick up the fork"
    }
    action_chunk = pi0.generate_action(example)
    print("Action chunk generated:", action_chunk)

def test_load_panda():
    model = mj.MjModel.from_xml_path(param.param.conveyor_system_xml)
    data = mj.MjData(model)
    panda = Panda(model, data)
    print("Arm joint positions:", panda.get_arm_joint_positions())
    print("Gripper positions:", panda.get_gripper_position())

if __name__ == "__main__":
    test_load_p0()