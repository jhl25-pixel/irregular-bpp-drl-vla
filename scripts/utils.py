import mujoco as mj
import mujoco.viewer as viewer

import sys
import os
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.param import param

def return_observation(xml_path : str) -> defaultdict:
    
    
