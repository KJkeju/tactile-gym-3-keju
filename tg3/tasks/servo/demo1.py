import numpy as np
import cv2
import os
from tg3.data.collect.setup_embodiment import setup_embodiment

def setup_environment(target_indicator=None, sensor_type='sim'):
    env_params = {
        'robot': 'sim_ur',
        'stim_name': 'cube',
        'speed': 50,
        'work_frame': (600, 0, 200, 180, 0, 180),
        'tcp_pose': (81, 0, 0, 0, 0, 0),
        'stim_pose': (350, -300, 60, 0, 0, 0),
        'show_gui': True,
        'load_target': target_indicator
    }
    sensor_params = {
        "type": sensor_type,
        "sensor_type": "right_angle_tactip",
        "image_size": (256, 256),
        "show_tactile": False
    }
    return setup_embodiment(env_params, sensor_params)

def collect_data(num_steps=100, save_dir="./collected_data"):
    os.makedirs(save_dir, exist_ok=True)
    robot, sensor = setup_environment()
    for step in range(num_steps):
        # 你可以随机、规律或者读取轨迹控制机械臂
        pose = np.array([300 + step, -300, 100, 0, 0, 0])  # 例：直线运动
        robot.move_linear(pose)
        tactile_img = sensor.process()
        cv2.imwrite(os.path.join(save_dir, f"tactile_{step:04d}.png"), tactile_img)
        # 保存机械臂位姿
        with open(os.path.join(save_dir, f"pose_{step:04d}.txt"), "w") as f:
            f.write(" ".join(map(str, pose.tolist())))
        print(f"Step {step}: Data saved.")

if __name__ == "__main__":
    collect_data(num_steps=200, save_dir="./collected_data")
