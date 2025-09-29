#!/usr/bin/env python3
"""
简单的MuJoCo XML查看器
使用方法: python G:\irregularBPP\env\viewer.py G:\irregularBPP\env\conveyor_system.xml
"""

import sys
import mujoco
import mujoco.viewer
import time
import cv2
import numpy as np


import utils
import param
import conveyor

class Images:

    def __init__(self):
        self.images = []
    
    def __len__(self):
        return len(self.images)

    def save_the_image(self, func):

        def wrapper(*args, **kwargs):

            self.images.append(func(*args, **kwargs))
        
        return wrapper

    def append(self, img):
        self.images.append(img)
    
    def return_img(self):
        return self.images

def save_image_to_video(images, output_path, fps=30):

    """
    将numpy图片列表保存为视频
    
    Args:
        images: list of numpy arrays, 每个array shape为(H, W, 3)
        output_path: 输出视频文件路径，如'output.mp4'
        fps: 帧率
    """
    if len(images) == 0:
        return
    
    # 获取图片尺寸
    height, width, channels = images[0].shape
    
    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for img in images:
        # OpenCV使用BGR格式，如果你的图片是RGB，需要转换
        if channels == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
            
        # 确保数据类型是uint8
        if img_bgr.dtype != np.uint8:
            img_bgr = (img_bgr * 255).astype(np.uint8) if img_bgr.max() <= 1.0 else img_bgr.astype(np.uint8)
        
        out.write(img_bgr)
    
    out.release()
    print(f"视频已保存到: {output_path}")

def view_xml(xml_path):
    """简单查看XML文件"""
    
    # 加载模型
    print(f"加载模型: {xml_path}")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 启动查看器
    print("启动3D查看器...")
    print("操作提示:")
    print("- 鼠标左键拖拽: 旋转视角")
    print("- 鼠标右键拖拽: 平移视角")
    print("- 鼠标滚轮: 缩放")
    print("- 按ESC键退出")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 运行仿真
        while viewer.is_running():
            step_start = time.time()
            
            
            # 清除之前的外力
            data.xfrc_applied[:] = 0
            
            # 对传送带上的物体施加力
            
            conveyor.apply_conveyor_velocity_simple(
                model, data, param.param.conveyor_xml, conveyor_speed=param.param.conveyor_speed
            )
            
            
            # 仿真一步
            mujoco.mj_step(model, data)
            
            # 同步到查看器
            viewer.sync()
            
            # 控制帧率
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

def view_env(xml_file):
    tmp_xml_file, cnt = utils.remove_outlier(xml_file)
    view_xml(tmp_xml_file)

def view_robot(xml_file):
    view_xml(xml_file)

def non_head_viewer(xml_path, camera_name='front_cam'):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera_name)
    
    img = renderer.render()
    import cv2
    cv2.imwrite("/home/hljin/irregular-bpp-drl-vla/img/sasasdads.png", img)

if __name__ == "__main__":
    '''
    E:/Install_Location/anaconda3/envs/mujoco/python.exe g:/irregularBPP/env/viewer.py
    '''

    #view_robot(r"G:\irregularBPP\experiment\2\scene_1.xml")
    #view_robot(param.param.conveyor_system_xml)

    non_head_viewer("/home/hljin/irregular-bpp-drl-vla/experiment/3/scene_4.xml")