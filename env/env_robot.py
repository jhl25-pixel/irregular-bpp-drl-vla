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

    def add_object_to_packing_region():
        
        tree = ET.parse(self.current_xml_path)
        root = tree.getroot()
        root.find()