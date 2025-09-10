import xml.etree.ElementTree as ET
import numpy as np
from typing import List

import utils

def set_the_connection_layer(xml_path, location : List[float], size : List[float]):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    for geom in worldbody.findall("geom"):
        if geom.get("name") == "connection_bridge":
            geom.set("size", f"{size[0]} {size[1]} {size[2]}")
            geom.set("pos", f"{location[0]} {location[1]} {location[2]}")
    
    tree.write(xml_path)

def set_the_size_and_location_of_the_host(xml_path, location : List[float], size : List[float]):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    #高度和z轴位置必须一样
    assert location[2] == size[2]

    for geom in worldbody.findall("geom"):
        if geom.get("name") == "warehouse_floor":
            geom.set("size", f"{size[0]} {size[1]} 0.15")
            geom.set("pos", f"{location[0]} {location[1]} 0.05")

        if geom.get("name") == "warehouse_back":
            geom.set("size", f"0.05 {size[1]} {size[2]}")
            geom.set("pos", f"{location[0] - size[0]} {location[1]} {location[2]}")

        if geom.get("name") == "warehouse_left":
            geom.set("size", f"{size[0]} 0.05 {size[2]}")
            geom.set("pos", f"{location[0]} {location[1]  - size[1]} {location[2]}")

        if geom.get("name") == "warehouse_right":
            geom.set("size", f"{size[0]} 0.05 {size[2]}")
            geom.set("pos", f"{location[0]} {location[1]  + size[1]} {location[2]}")

        if geom.get("name") == "warehouse_roof":
            geom.set("size", f"{size[0]} {size[1]} 0.05")
            geom.set("pos", f"{location[0]} {location[1]} 0.75")
    
    tree.write(xml_path)

def set_the_size_and_location_of_the_conveyor(xml_path: str, size=[1.5, 0.25, 0.08], location=[0, -0.5, 0.2]):
    '''
    设置传送带尺寸
    size and location
    xpos: 
    '''

    x_size, y_size, z_size = size[0], size[1], size[2]
    xpos, ypos, zpos = location[0], location[1], location[2]

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")



    for geom in worldbody.findall("geom"):
        name = geom.get("name")
        
        if name == "left_support":
            # 解析当前位置
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                # 设置新位置：X = -x_size, 保持Y和Z不变
                geom.set("pos", f"{xpos-x_size} {ypos} {pos[2]}")
                # 设置尺寸：支撑柱尺寸
                geom.set("size", f"0.1 0.3 {z_size}")
        
        elif name == "right_support":
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{xpos + x_size} {ypos} {pos[2]}")
                geom.set("size", f"0.1 0.3 {z_size}")
        
        elif name == "frame_base":
            size_str = geom.get("size")
            if size_str:
                size = [float(x) for x in size_str.split()]
                # 框架底座X尺寸 = x_size + 0.1
                geom.set("size", f"{x_size + 0.1} {size[1]} {size[2]}")
                # 调整底座高度位置
                pos_str = geom.get("pos")
                if pos_str:
                    pos = [float(x) for x in pos_str.split()]
                    geom.set("pos", f"{xpos} {ypos} {z_size/2}")

        elif name == "belt_surface":
            # 传送带表面尺寸
            geom.set("size", f"{x_size} {y_size} {z_size}")
            # 调整表面高度位置：Z = z_size + 表面厚度/2
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{xpos} {ypos} {z_size + 0.08/2}")

        elif name == "left_rail":
            # 护栏尺寸和位置
            geom.set("size", f"{x_size} 0.02 0.05")
            # Y坐标 = 中心Y - y_size - 0.02
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{xpos} {ypos - y_size - 0.02} {z_size + 0.05}")
        
        elif name == "right_rail":
            geom.set("size", f"{x_size} 0.02 0.05")
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{xpos} {ypos - y_size + 0.02} {z_size + 0.05}")

    # 保存修改
    tree.write(xml_path)
    print(f"传送带尺寸已更新: x_size={x_size}, y_size={y_size}, z_size={z_size}")

def set_the_size_of_the_conveyor(xml_path: str, size=[3, 0.25, 0.08], location=[0, -0,5, 0,2]):
    '''
    设置传送带尺寸
    x_size: 传送带长度的一半
    y_size: 传送带宽度的一半  
    z_size: 传送带高度的一半（支撑柱高度）
    xpos: 
    '''
    x_size, y_size, z_size = size[0], size[1], size[2]
    xpos, ypos, zpos = location[0], location[1], location[2]
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    for geom in worldbody.findall("geom"):
        name = geom.get("name")
        
        if name == "left_support":
            # 解析当前位置
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                # 设置新位置：X = -x_size, 保持Y和Z不变
                geom.set("pos", f"{xpos-x_size} {ypos} {pos[2]}")
                # 设置尺寸：支撑柱尺寸
                geom.set("size", f"0.1 0.3 {z_size}")
        
        elif name == "right_support":
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{xpos + x_size} {ypos} {pos[2]}")
                geom.set("size", f"0.1 0.3 {z_size}")
        
        elif name == "frame_base":
            size_str = geom.get("size")
            if size_str:
                size = [float(x) for x in size_str.split()]
                # 框架底座X尺寸 = x_size + 0.1
                geom.set("size", f"{x_size + 0.1} {size[1]} {size[2]}")
                # 调整底座高度位置
                pos_str = geom.get("pos")
                if pos_str:
                    pos = [float(x) for x in pos_str.split()]
                    geom.set("pos", f"{xpos} {ypos} {z_size/2}")

        elif name == "belt_surface":
            # 传送带表面尺寸
            geom.set("size", f"{x_size} {y_size} {z_size}")
            # 调整表面高度位置：Z = z_size + 表面厚度/2
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{xpos} {ypos} {z_size + 0.08/2}")

        elif name == "left_rail":
            # 护栏尺寸和位置
            geom.set("size", f"{x_size} 0.02 0.05")
            # Y坐标 = 中心Y - y_size - 0.02
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{xpos} {ypos - y_size - 0.02} {z_size + 0.05}")
        
        elif name == "right_rail":
            geom.set("size", f"{x_size} 0.02 0.05")
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{xpos} {ypos - y_size + 0.02} {z_size + 0.05}")

    # 保存修改
    tree.write(xml_path)
    print(f"传送带尺寸已更新: x_size={x_size}, y_size={y_size}, z_size={z_size}")

def apply_conveyor_velocity_simple(model, data, xml_path, conveyor_speed=1.0):
    """
    简化版本：直接将传送带上的物体速度设置为传送带速度
    """
    # 获取传送带范围
    conveyor_min, conveyor_max = utils.return_conveyor_range_worldbody(xml_path)
    host_min, host_max = utils.return_host_range_worldbody(xml_path)
    bridge_min, bridge_max = utils.return_connector_range_worldbody(xml_path)
    # 传送带运动方向（假设沿X轴正方向）
    conveyor_velocity = np.array([conveyor_speed, 0.0, 0.0])
    
    # 遍历所有物体
    for body_id in range(1, model.nbody):
        # 获取物体重心位置
        body_pos = data.xpos[body_id].copy()
        
        
            
        # 检查是否在传送带上
        if utils.is_on_conveyor(body_pos, conveyor_min, conveyor_max) or utils.is_on_host(body_pos, host_min, host_max) or utils.is_on_bridge(body_pos, bridge_min, bridge_max):
            # 获取物体的自由度索引
            body_dofadr = model.body_dofadr[body_id]
            body_dofnum = model.body_dofnum[body_id]
            
            if body_dofnum >= 3:  # 确保至少有3个线性自由度
                # 直接设置X方向速度为传送带速度，保持Y、Z速度不变
                data.qvel[body_dofadr] = conveyor_speed  # X方向




def set_up_the_conveyor_system(xml_path, conveyor_size=[3, 0.25, 0.08], location=[-1.5, -0.5, 0.2], connection_length=0.25,
                               host_size=[0.8, 0.6, 0.4]):
    '''
    size : the half size of the conveyor
    location : the center location of the conveyor
    '''
    set_the_size_and_location_of_the_conveyor(xml_path, size=conveyor_size, location=location)
    connection_layer_x_center = location[0] - conveyor_size[0] - connection_length
    connection_layer_y_center = location[1]
    set_the_connection_layer(xml_path, size=[connection_length, 0.25, 0.02], location=[connection_layer_x_center, connection_layer_y_center, 0.18])
    host_x_center = connection_layer_x_center - connection_length - host_size[0]
    host_y_center = location[1]
    set_the_size_and_location_of_the_host(xml_path, size=host_size, location=[host_x_center, host_y_center, host_size[2]])


if __name__ == "__main__":
    #set_the_size_and_location_of_the_conveyor(r"G:\irregularBPP\env\franka_emika_panda\conveyor.xml", x_size=3, xpos=-1.5)
    #set_the_size_and_location_of_the_host(r"G:\irregularBPP\env\franka_emika_panda\conveyor.xml", )
    set_up_the_conveyor_system(r"G:\irregularBPP\env\franka_emika_panda\conveyor.xml",
                                conveyor_size=[1, 0.25, 0.08],
                                location=[-0, -0.5, 0.2],
                                connection_length=0.25,
                                host_size=[3.8, 0.6, 0.4])