import os
import datetime

class param:
    obj_num=5 # state
    hidden_dim = 128
    seed=42
    obj_scale_target_volume = 280
    scale=0.003
    absolute_irbpp_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    robot_xml = os.path.join(absolute_irbpp_root_path, "env", "franka_emika_panda", "panda.xml")
    robot_assets = os.path.join(absolute_irbpp_root_path, "env", "franka_emika_panda", "assets")
    conveyor_xml = os.path.join(absolute_irbpp_root_path, "env", "franka_emika_panda", "conveyor.xml")
    data_path = os.path.join(absolute_irbpp_root_path, "dataset", "objaversestl")
    xml_path = os.path.join(absolute_irbpp_root_path, "env", "franka_emika_panda")
    result_path = os.path.join(absolute_irbpp_root_path, "experiment")
    data_num=100
    required_stable_steps=50
    stable_threshold_v = 1e-4
    time = None
    res_idx = 2
    result_path_now = None
    epoches = data_num - 1
    def __init__(self, seed=42, data_path = "../dataset/objaversestl"):

        self.seed = seed
        self.data_path = data_path
    @classmethod
    def get_seed(cls):
        return cls.seed

    @classmethod
    def get_data_path(cls):
        return cls.data_path

if __name__ == "__main__":
    absolute_irbpp_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(absolute_irbpp_root_path, "dataset", "objaversestl")
    print(data_path)