import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
import torch
from collections import deque
from param import param
class N_OBJ_State:
    def __init__(self, xml_path, obj_num=param.obj_num):
        self.xml_path = xml_path
        self.obj_num = obj_num
        self.model = None
        self.data = None
        
        # 直接存储数值而不是字典，更高效
        self.state_queue = deque(maxlen=obj_num)
        self.reset()
    
    def reset(self):
        """重置状态"""
        self.state_queue.clear()
        # 初始化空状态（7个0）
        empty_state = np.zeros(7, dtype=np.float32)
        for _ in range(self.obj_num):
            self.state_queue.append(empty_state.copy())
    
    def return_the_state(self):
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        """获取状态tensor"""

        mj.mj_forward(self.model, self.data)
        # 获取当前物体状态
        current_states = self._get_current_states_array()
        
        # 更新队列：头部插入新状态
        for state in reversed(current_states):
            self.state_queue.appendleft(state)
        
        # 拼接所有状态
        all_states = np.concatenate(list(self.state_queue), axis=0)
        
        # 转换为tensor
        return torch.FloatTensor(all_states)
    
    def _get_current_states_array(self):
        """获取当前状态为numpy数组"""
        states = []
        
        for i in range(self.model.nbody):
            body_name = self.model.body(i).name
            if body_name and body_name.startswith("obj_"):
                pos = self.data.body(i).xpos.copy()
                quat = self.data.body(i).xquat.copy()
                
                # 直接创建7维数组
                state_array = np.array([
                    float(pos[0]), float(pos[1]), float(pos[2]),
                    float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                ], dtype=np.float32)
                
                states.append(state_array)
        
        return states
    
    def get_state_shape(self):
        """获取状态tensor的形状"""
        return (self.obj_num * 7,)

        





