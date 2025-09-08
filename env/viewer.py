#!/usr/bin/env python3
"""
简单的MuJoCo XML查看器
使用方法: python G:\irregularBPP\env\viewer.py G:\irregularBPP\env\conveyor_system.xml
"""

import sys
import mujoco
import mujoco.viewer
import time

import utils
import param
import conveyor

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
                model, data, param.param.conveyor_xml, conveyor_speed=0.3
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


if __name__ == "__main__":
    '''
    E:/Install_Location/anaconda3/envs/mujoco/python.exe g:/irregularBPP/env/viewer.py
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="G:\irregularBPP\experiment\1\scene_0.xml")
    args = parser.parse_args()
    #view_robot(r"G:\irregularBPP\experiment\2\scene_1.xml")
    view_robot(args.path)