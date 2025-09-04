import os
import random
from pathlib import Path
from stl import mesh

from param import param
import utils

class simulator:

    def __init__(self, file_path : str, data_type="stl"):
        
        "创造一个对象构建器"
        self.file_path = file_path
        self.data_type = data_type

    def _load_the_data_list(self, 
        data_type : str,
    ):
        item_name_list = os.listdir(self.file_path)
        data_path_list = [ os.path.join(self.file_path, item) for item in item_name_list]
        valid_data_path_list = self._obtain_the_valid_data(data_path_list)
        return valid_data_path_list

    def _obtain_the_valid_data(self, data_path_list):
        valid_data_path_list = []
        for item in data_path_list:
            try:
                ascii_mesh = mesh.Mesh.from_file(item)
                utils.convert_stl_from_ascii_to_binary(item, item)
            except:
                continue

            valid_data_path_list.append(item)
        return valid_data_path_list

    def _roll_the_dice(self, roll_num: int = 50):
        
        source_object_list = self._load_the_data_list(self.data_type)
        simulated_object_list = random.choices(source_object_list, k = roll_num)
        
        return simulated_object_list
    
if __name__ == "__main__":

    data_path = param.data_path
    simulator = simulator(data_path)
    print(simulator._roll_the_dice(200))
    
