import xml.etree.ElementTree as ET
import numpy as np
import os
import mujoco as mj
import random

from simulator import simulator
from param import param
from utils import is_binary_stl


'''
-> from up to down rebuild a mujoco environment with scalable collection box
'''
class MuJoCoEnvironmentInitializer:
    
    def __init__(self, object_num=50, box_scale=1.0, box_position=(0, 0, 0.1),
                 conveyor_length=2.0, conveyor_width=0.5, conveyor_height=0.1,
                 conveyor_position=(0, 0, 0), conveyor_speed=0.5):
        """
        初始化MuJoCo环境
        
        Args:
            object_num: 物体数量
            box_scale: 盒子缩放因子 (1.0=原始大小, 0.5=一半, 2.0=两倍)
            box_position: 盒子位置 (x, y, z)
            conveyor_length: 传送带长度
            conveyor_width: 传送带宽度
            conveyor_height: 传送带高度
            conveyor_position: 传送带位置 (x, y, z)
            conveyor_speed: 传送带速度
        """
        self.root = None
        self.worldbody = None
        self.object_num = object_num
        self.object_cnt = 0
        self.box_scale = box_scale
        self.box_position = box_position
        self.conveyor_length = conveyor_length
        self.conveyor_width = conveyor_width
        self.conveyor_height = conveyor_height
        self.conveyor_position = conveyor_position
        self.conveyor_speed = conveyor_speed
        
        # 基础尺寸定义（将被缩放因子调整）
        self.base_box_size = {
            'bottom': (1.5, 1.5, 0.05),  # 底面尺寸
            'wall_thickness': 0.05,       # 墙壁厚度
            'wall_height': 0.8,           # 墙壁高度
        }

    def generate_xml(self, filename: str = "conveyor_system.xml"):
        """生成修正的XML文件"""
        
        # 创建根元素
        self.root = ET.Element("mujoco", model="conveyor_box_system")
        self._add_franka_robot_and_conveyor()
        # 编译器设置
        compiler = ET.SubElement(self.root, "compiler")
        compiler.set("angle", "radian")
        compiler.set("coordinate", "local")
        compiler.set("inertiafromgeom", "true")
        
        # 仿真选项
        option = ET.SubElement(self.root, "option")
        option.set("timestep", "0.001")
        option.set("gravity", "0 0 -9.81")
        option.set("integrator", "RK4")
        
        # 添加材质
        self._add_assets()
        
        # 创建世界主体
        self._create_world()

        # 添加执行器
        #self._add_actuators()
        
        # 添加传感器
        #self._add_sensors()
        
        
        # 格式化并保存
        self._indent_xml(self.root)
        tree = ET.ElementTree(self.root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        print(f"✓ XML文件已生成: {filename}")
        print(f"✓ 盒子缩放因子: {self.box_scale}")
        print(f"✓ 盒子位置: {self.box_position}")

        return filename
    
    def _add_default_information(self):

        default = ET.SubElement(self.root, "default")
        

    def _add_franka_robot_and_conveyor(self):


        include = ET.SubElement(self.root, "include")
        include.set("file", param.robot_xml)

        include2 = ET.SubElement(self.root, "include")
        include2.set("file", param.conveyor_xml)

    def _add_assets(self):
        """添加材质和资源"""
        asset = ET.SubElement(self.root, "asset")
        
        # 材质定义
        materials = {
            "box_material": "0.8 0.6 0.4 1",
            "ground_material": "0.9 0.9 0.9 1",
            "wall_material": "0.7 0.7 0.8 1",
        }
        
        for name, rgba in materials.items():
            material = ET.SubElement(asset, "material")
            material.set("name", name)
            material.set("rgba", rgba)
        
        # 纹理
        texture = ET.SubElement(asset, "texture")
        texture.set("name", "grid")
        texture.set("type", "2d")
        texture.set("builtin", "checker")
        texture.set("rgb1", "0.1 0.2 0.3")
        texture.set("rgb2", "0.2 0.3 0.4")
        texture.set("width", "300")
        texture.set("height", "300")
        
        grid_material = ET.SubElement(asset, "material")
        grid_material.set("name", "grid_material")
        grid_material.set("texture", "grid")
        grid_material.set("texrepeat", "8 8")
        grid_material.set("rgba", "0.8 0.8 0.8 1")
    
    def _get_the_object_asset_from_simulator(self):
        mujuco_object_simulator = simulator(param.get_data_path())
        data_path_list = mujuco_object_simulator._roll_the_dice(self.object_num)
        data_path_list_unique = list(set(data_path_list))
        return data_path_list_unique, data_path_list

    def _create_world(self):
        """创建世界内容"""
        self.worldbody = ET.SubElement(self.root, "worldbody")
        
        # 地面 - 作为world的直接子元素，根据盒子大小适当调整地面大小
        ground_size = max(10, self.box_scale * 5)  # 确保地面足够大
        ground = ET.SubElement(self.worldbody, "geom")
        ground.set("name", "ground")
        ground.set("type", "plane")
        ground.set("size", f"{ground_size} {ground_size} 0.1")
        ground.set("material", "grid_material")
        
        # 收集箱
        self._add_collection_box()
        

        
        # 光源
        self._add_lighting()
    
    def _add_collection_box(self):
        """添加可缩放的收集箱"""
        box_body = ET.SubElement(self.worldbody, "body")
        box_body.set("name", "collection_box")
        
        # 应用缩放后的位置
        scaled_pos = (
            self.box_position[0] * self.box_scale,
            self.box_position[1] * self.box_scale,
            self.box_position[2] * self.box_scale
        )
        box_body.set("pos", f"{scaled_pos[0]} {scaled_pos[1]} {scaled_pos[2]}")
        
        # 计算缩放后的尺寸
        base_width = self.base_box_size['bottom'][0]
        base_length = self.base_box_size['bottom'][1]
        base_thickness = self.base_box_size['bottom'][2]
        wall_thickness = self.base_box_size['wall_thickness']
        wall_height = self.base_box_size['wall_height']
        
        # 应用缩放
        scaled_width = base_width * self.box_scale
        scaled_length = base_length * self.box_scale
        scaled_thickness = base_thickness * self.box_scale
        scaled_wall_thickness = wall_thickness * self.box_scale
        scaled_wall_height = wall_height * self.box_scale
        
        # 箱子底面
        bottom = ET.SubElement(box_body, "geom")
        bottom.set("name", "box_bottom")
        bottom.set("type", "box")
        bottom.set("size", f"{scaled_width} {scaled_length} {scaled_thickness}")
        bottom.set("material", "box_material")
        
        # 计算墙壁位置（考虑缩放）
        wall_offset = scaled_width - scaled_wall_thickness
        wall_center_height = scaled_thickness + scaled_wall_height / 2
        wall_inner_length = scaled_length - 2 * scaled_wall_thickness
        
        # 箱子四面墙（按缩放因子调整所有尺寸和位置）
        walls = [
            # (name, position, size)
            ("box_wall1", f"0 {wall_offset} {wall_center_height}", 
             f"{scaled_width} {scaled_wall_thickness} {scaled_wall_height}"),
            ("box_wall2", f"0 {-wall_offset} {wall_center_height}", 
             f"{scaled_width} {scaled_wall_thickness} {scaled_wall_height}"),
            ("box_wall3", f"{wall_offset} 0 {wall_center_height}", 
             f"{scaled_wall_thickness} {wall_inner_length} {scaled_wall_height}"),
            ("box_wall4", f"{-wall_offset} 0 {wall_center_height}", 
             f"{scaled_wall_thickness} {wall_inner_length} {scaled_wall_height}")
        ]
        
        for name, pos, size in walls:
            wall = ET.SubElement(box_body, "geom")
            wall.set("name", name)
            wall.set("type", "box")
            wall.set("size", size)
            wall.set("pos", pos)
            wall.set("material", "wall_material")
    
    def _add_conveyor_system(self):
        """添加传送带系统 - 静态版本"""
        # 传送带主体
        conveyor_body = ET.SubElement(self.worldbody, "body")
        conveyor_body.set("name", "conveyor")
        conveyor_body.set("pos", f"{self.conveyor_position[0]} {self.conveyor_position[1]} {self.conveyor_position[2]}")
        
        # 传送带平台
        conveyor_platform = ET.SubElement(conveyor_body, "geom")
        conveyor_platform.set("name", "conveyor_platform")
        conveyor_platform.set("type", "box")
        conveyor_platform.set("size", f"{self.conveyor_length/2} {self.conveyor_width/2} {self.conveyor_height/2}")
        conveyor_platform.set("pos", "0 0 0")
        conveyor_platform.set("material", "conveyor_material")
        
        # 传送带侧面挡板
        side_wall_height = self.conveyor_height * 1.5
        side_wall_thickness = 0.02
        
        # 左侧挡板
        left_wall = ET.SubElement(conveyor_body, "geom")
        left_wall.set("name", "conveyor_left_wall")
        left_wall.set("type", "box")
        left_wall.set("size", f"{self.conveyor_length/2} {side_wall_thickness/2} {side_wall_height/2}")
        left_wall.set("pos", f"0 {-self.conveyor_width/2 - side_wall_thickness/2} {side_wall_height/2 - self.conveyor_height/2}")
        left_wall.set("material", "wall_material")
        
        # 右侧挡板
        right_wall = ET.SubElement(conveyor_body, "geom")
        right_wall.set("name", "conveyor_right_wall")
        right_wall.set("type", "box")
        right_wall.set("size", f"{self.conveyor_length/2} {side_wall_thickness/2} {side_wall_height/2}")
        right_wall.set("pos", f"0 {self.conveyor_width/2 + side_wall_thickness/2} {side_wall_height/2 - self.conveyor_height/2}")
        right_wall.set("material", "wall_material")
        
        # 传送带滚轮 - 作为静态几何体
        roller_radius = 0.05
        roller_length = self.conveyor_width
        num_rollers = 6
        
        for i in range(num_rollers):
            x_pos = -self.conveyor_length/2 + (i + 0.5) * (self.conveyor_length / num_rollers)
            
            # 直接在主体上添加滚轮几何体
            roller_geom = ET.SubElement(conveyor_body, "geom")
            roller_geom.set("name", f"roller_geom_{i}")
            roller_geom.set("type", "cylinder")
            roller_geom.set("size", f"{roller_radius} {roller_length/2}")
            roller_geom.set("pos", f"{x_pos} 0 {roller_radius}")
            roller_geom.set("material", "roller_material")
    

    
    def get_box_info(self):
        """获取当前盒子的信息"""
        scaled_width = self.base_box_size['bottom'][0] * self.box_scale
        scaled_length = self.base_box_size['bottom'][1] * self.box_scale
        scaled_height = (self.base_box_size['bottom'][2] + self.base_box_size['wall_height']) * self.box_scale
        
        return {
            'scale_factor': self.box_scale,
            'position': self.box_position,
            'dimensions': {
                'width': scaled_width * 2,  # 总宽度
                'length': scaled_length * 2,  # 总长度
                'height': scaled_height,  # 总高度
                'inner_width': (scaled_width - self.base_box_size['wall_thickness'] * self.box_scale) * 2,
                'inner_length': (scaled_length - self.base_box_size['wall_thickness'] * self.box_scale) * 2,
                'volume': (scaled_width * 2) * (scaled_length * 2) * scaled_height
            },
            'conveyor': {
                'length': self.conveyor_length,
                'width': self.conveyor_width,
                'height': self.conveyor_height,
                'position': self.conveyor_position,
                'speed': self.conveyor_speed
            }
        }
    
    def _add_lighting(self):
        """添加光照（根据场景大小调整光源位置）"""
        # 主光源 - 根据缩放调整高度
        light_height = max(10, self.box_scale * 5)
        main_light = ET.SubElement(self.worldbody, "light")
        main_light.set("name", "main_light")
        main_light.set("pos", f"0 0 {light_height}")
        main_light.set("dir", "0 0 -1")
        main_light.set("diffuse", "0.8 0.8 0.8")
        main_light.set("specular", "0.2 0.2 0.2")
        
        # 侧光源 - 根据缩放调整位置
        side_distance = max(5, self.box_scale * 3)
        side_light = ET.SubElement(self.worldbody, "light")
        side_light.set("name", "side_light")
        side_light.set("pos", f"{side_distance} {side_distance} {side_distance}")
        side_light.set("dir", "-1 -1 -1")
        side_light.set("diffuse", "0.4 0.4 0.4")
        
    
    def _add_actuators(self):
        """添加执行器"""
        actuator = ET.SubElement(self.root, "actuator")
        
        # 传送带滚轮执行器
        num_rollers = 6
        for i in range(num_rollers):
            motor = ET.SubElement(actuator, "motor")
            motor.set("name", f"roller_motor_{i}")
            motor.set("joint", f"roller_joint_{i}")
            motor.set("gear", "100")
            motor.set("ctrllimited", "true")
            motor.set("ctrlrange", f"{-self.conveyor_speed*10} {self.conveyor_speed*10}")
    
    def _add_sensors(self):
        """添加传感器"""
        sensor = ET.SubElement(self.root, "sensor")
        
        # 传送带滚轮传感器
        num_rollers = 6
        for i in range(num_rollers):
            # 位置传感器
            pos_sensor = ET.SubElement(sensor, "jointpos")
            pos_sensor.set("name", f"roller_{i}_pos")
            pos_sensor.set("joint", f"roller_joint_{i}")
            
            # 速度传感器
            vel_sensor = ET.SubElement(sensor, "jointvel")
            vel_sensor.set("name", f"roller_{i}_vel")
            vel_sensor.set("joint", f"roller_joint_{i}")

    
    
    def _indent_xml(self, elem, level=0):
        """格式化XML缩进"""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent_xml(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

def generate_initial_xml(box_scale=1.0, box_position=(0, 0, 0.1), object_num=50,
                        conveyor_length=2.0, conveyor_width=0.5, conveyor_height=0.1,
                        conveyor_position=(0, 0, 0), conveyor_speed=0.5):
    """
    生成可缩放盒子的XML文件
    
    Args:
        box_scale: 盒子缩放因子
        box_position: 盒子位置
        object_num: 物体数量
        conveyor_length: 传送带长度
        conveyor_width: 传送带宽度
        conveyor_height: 传送带高度
        conveyor_position: 传送带位置
        conveyor_speed: 传送带速度
    """
    generator = MuJoCoEnvironmentInitializer(
        object_num=object_num,
        box_scale=box_scale, 
        box_position=box_position,
        conveyor_length=conveyor_length,
        conveyor_width=conveyor_width,
        conveyor_height=conveyor_height,
        conveyor_position=conveyor_position,
        conveyor_speed=conveyor_speed
    )
    xml_path = os.path.join(param.xml_path, "conveyor_system.xml")
    filename = generator.generate_xml(xml_path)
    
    # 打印场景信息
    scene_info = generator.get_box_info()
    print("\n📦 场景信息:")
    print(f"   盒子缩放因子: {scene_info['scale_factor']}")
    print(f"   盒子尺寸: {scene_info['dimensions']['width']:.2f} × {scene_info['dimensions']['length']:.2f} × {scene_info['dimensions']['height']:.2f}")
    print(f"   传送带尺寸: {scene_info['conveyor']['length']:.2f} × {scene_info['conveyor']['width']:.2f} × {scene_info['conveyor']['height']:.2f}")
    print(f"   传送带位置: {scene_info['conveyor']['position']}")
    print(f"   传送带速度: {scene_info['conveyor']['speed']:.2f}")
    print(f"   物体数量: {object_num}")
    
    print("\n✓ 完整场景XML文件生成完成!")
    print(f"✓ 文件位置: {filename}")
    
    return filename

# 便捷函数用于生成不同配置的场景
def generate_default_scene():
    """生成默认场景"""
    return generate_initial_xml()

def generate_large_scene():
    """生成大场景"""
    return generate_initial_xml(
        box_scale=1.5, 
        box_position=(3, 0, 0.1),
        conveyor_length=3.0,
        conveyor_width=0.7,
        object_num=30
    )

def generate_small_scene():
    """生成小场景"""
    return generate_initial_xml(
        box_scale=0.7, 
        box_position=(2, 0, 0.1),
        conveyor_length=1.5,
        conveyor_width=0.4,
        object_num=20
    )


if __name__ == "__main__":
    # 生成标准场景
    generate_initial_xml(
        box_scale=0.1, 
        box_position=(5, 0, 0.1),
        conveyor_length=2.5,
        conveyor_width=0.6,
        conveyor_position=(-10, 10, 0.5),
        object_num=25
    )
    
    # 查看生成的场景
    from viewer import view_xml
    view_xml(os.path.join(param.xml_path, "conveyor_system.xml"))