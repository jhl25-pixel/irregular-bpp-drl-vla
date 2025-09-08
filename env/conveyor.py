import xml.etree.ElementTree as ET
import numpy as np

import utils

def set_the_size_and_location_of_the_conveyor(xml_path: str, x_size=1.5, y_size=0.25, z_size=0.08,
                                 xpos=0, ypos=-0.5):
    '''
    设置传送带尺寸
    x_size: 传送带长度的一半
    y_size: 传送带宽度的一半  
    z_size: 传送带高度的一半（支撑柱高度）
    xpos: 
    '''
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

def set_the_size_of_the_conveyor(xml_path: str, x_size=1.5, y_size=0.25, z_size=0.08,
                                 xpos=0, ypos=-0.5, zpos=0.2):
    '''
    设置传送带尺寸
    x_size: 传送带长度的一半
    y_size: 传送带宽度的一半  
    z_size: 传送带高度的一半（支撑柱高度）
    xpos: 
    '''
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
                geom.set("pos", f"{-x_size} {pos[1]} {pos[2]}")
                # 设置尺寸：支撑柱尺寸
                geom.set("size", f"0.1 0.3 {z_size}")
        
        elif name == "right_support":
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{x_size} {pos[1]} {pos[2]}")
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
                    geom.set("pos", f"{pos[0]} {pos[1]} {z_size/2}")

        elif name == "belt_surface":
            # 传送带表面尺寸
            geom.set("size", f"{x_size} {y_size} {z_size}")
            # 调整表面高度位置：Z = z_size + 表面厚度/2
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{pos[0]} {pos[1]} {z_size + 0.08/2}")

        elif name == "left_rail":
            # 护栏尺寸和位置
            geom.set("size", f"{x_size} 0.02 0.05")
            # Y坐标 = 中心Y - y_size - 0.02
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{pos[0]} {-0.5 - y_size - 0.02} {z_size + 0.05}")
        
        elif name == "right_rail":
            geom.set("size", f"{x_size} 0.02 0.05")
            pos_str = geom.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                geom.set("pos", f"{pos[0]} {-0.5 - y_size + 0.02} {z_size + 0.05}")

    # 保存修改
    tree.write(xml_path)
    print(f"传送带尺寸已更新: x_size={x_size}, y_size={y_size}, z_size={z_size}")

def apply_conveyor_velocity_simple(model, data, xml_path, conveyor_speed=1.0):
    """
    简化版本：直接将传送带上的物体速度设置为传送带速度
    """
    # 获取传送带范围
    conveyor_min, conveyor_max = utils.return_conveyor_range_worldbody(xml_path)
    
    # 传送带运动方向（假设沿X轴正方向）
    conveyor_velocity = np.array([conveyor_speed, 0.0, 0.0])
    
    # 遍历所有物体
    for body_id in range(1, model.nbody):
        # 获取物体重心位置
        body_pos = data.xpos[body_id].copy()
        
        if model.body(body_id).name == "obj_1" and not utils.is_on_conveyor(body_pos, conveyor_min, conveyor_max):
            continue
            
        # 检查是否在传送带上
        if utils.is_on_conveyor(body_pos, conveyor_min, conveyor_max):
            # 获取物体的自由度索引
            body_dofadr = model.body_dofadr[body_id]
            body_dofnum = model.body_dofnum[body_id]
            
            if body_dofnum >= 3:  # 确保至少有3个线性自由度
                # 直接设置X方向速度为传送带速度，保持Y、Z速度不变
                data.qvel[body_dofadr] = conveyor_speed  # X方向

def apply_conveyor_force(model, data, xml_path, conveyor_speed=1.0, force_magnitude=10.0):
    """
    对传送带上的物体施加恒定力
    conveyor_speed: 传送带速度（m/s）
    force_magnitude: 力的大小（N）
    """
    # 获取传送带范围
    conveyor_min, conveyor_max = utils.return_conveyor_range_worldbody(xml_path)
    
    # 传送带运动方向（假设沿X轴正方向）
    conveyor_direction = np.array([1.0, 0.0, 0.0])
    
    # 遍历所有物体
    for body_id in range(1, model.nbody):  # 跳过世界坐标系(id=0)
        # 获取物体重心位置
        body_pos = data.xpos[body_id].copy()
        if model.body(body_id).name == "obj_1" and not utils.is_on_conveyor(body_pos, conveyor_min, conveyor_max):
            1
        # 检查是否在传送带上
        if utils.is_on_conveyor(body_pos, conveyor_min, conveyor_max):
            # 获取物体当前速度
            body_vel = data.cvel[body_id][:3]  # 线速度部分
            
            # 计算相对速度（物体速度 - 传送带速度）
            relative_vel = body_vel - conveyor_speed * conveyor_direction
            
            # 计算摩擦力方向（与相对速度相反）
            if np.linalg.norm(relative_vel) > 1e-6:
                friction_direction = -relative_vel / np.linalg.norm(relative_vel)
            else:
                friction_direction = conveyor_direction
            
            # 施加力到物体重心
            force = force_magnitude * friction_direction
            
            # 在物体重心处施加外力
            data.xfrc_applied[body_id][:3] += force

if __name__ == "__main__":
    set_the_size_and_location_of_the_conveyor(r"G:\irregularBPP\env\franka_emika_panda\conveyor.xml", x_size=3, xpos=-1.5)