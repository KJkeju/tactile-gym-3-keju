import os

import ipdb

from tg3.simulator.assets import add_assets_path
from tg3.simulator.robots.arms import arm_mapping
from tg3.simulator.sensors.tactile_sensor import TactileSensor
from tg3.simulator.sensors.vision_sensor import VisionSensor


class ArmEmbodiment:
    def __init__(
        self,
        pb,
        robot_arm_params={}
    ):

        self._pb = pb
        self.arm_type = robot_arm_params["type"]

        if "tcp_link_name" in robot_arm_params:
            self.tcp_link_name = robot_arm_params["tcp_link_name"]
        else:
            self.tcp_link_name = "ee_link"

        # load the urdf file
        self.load_urdf()

        # instantiate a robot arm
        self.arm = arm_mapping[self.arm_type](
            pb,
            embodiment_id=self.embodiment_id,
            tcp_link_id=self.tcp_link_id,
            link_name_to_index=self.link_name_to_index,
            joint_name_to_index=self.joint_name_to_index,
            rest_poses=robot_arm_params['rest_poses'],
        )

    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()

    def load_urdf(self):
        """
        Load the robot arm model into pybullet
        """
        self.base_pos = [0, 0, 0]
        self.base_rpy = [0, 0, 0]
        self.base_orn = self._pb.getQuaternionFromEuler(self.base_rpy)
        asset_name = os.path.join(
            "robot_arm_assets",
            self.arm_type,
            "urdfs",
            self.arm_type + ".urdf",
        )
        print("\n[DEBUG] Will load URDF file:", add_assets_path(asset_name))

        self.embodiment_id = self._pb.loadURDF(
            add_assets_path(asset_name), self.base_pos, self.base_orn, useFixedBase=True
        )

        # create dicts for mapping link/joint names to corresponding indices
        self.num_joints, self.link_name_to_index, self.joint_name_to_index = self.create_link_joint_mappings(
            self.embodiment_id)

        # get the link and tcp IDs
        self.tcp_link_id = self.link_name_to_index[self.tcp_link_name]

        num_joints = self._pb.getNumJoints(self.embodiment_id)
        all_link_indices = [-1] + list(range(num_joints))

        for li in all_link_indices:
            self._pb.setCollisionFilterGroupMask(self.embodiment_id, li, 0, 0)

    def create_link_joint_mappings(self, urdf_id):

        num_joints = self._pb.getNumJoints(urdf_id)

        # pull relevent info for controlling the robot
        joint_name_to_index = {}
        link_name_to_index = {}
        for i in range(num_joints):
            info = self._pb.getJointInfo(urdf_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            joint_name_to_index[joint_name] = i
            link_name_to_index[link_name] = i

        return num_joints, link_name_to_index, joint_name_to_index

    def reset(self, reset_tcp_pose):
        """
        Reset the pose of the arm and sensor
        """
        self.arm.reset()

        # move to the initial position
        self.arm.move_linear(reset_tcp_pose, quick_mode=True)

    def full_reset(self):
        self.load_urdf()
        self.sensor.turn_off_collisions()


# class TactileArmEmbodiment(ArmEmbodiment):
#     def __init__(
#         self,
#         pb,
#         robot_arm_params={},
#         tactile_sensor_params={}
#     ):
#         self._pb = pb
#         self.arm_type = robot_arm_params["type"]
#         self.tactile_sensor_type = tactile_sensor_params["type"]
#
#         if "tcp_link_name" in robot_arm_params:
#             self.tcp_link_name = robot_arm_params["tcp_link_name"]
#         else:
#             self.tcp_link_name = "ee_link"
#
#         # load the urdf file
#         self.load_urdf()
#
#         # instantiate a robot arm
#         self.arm = arm_mapping[self.arm_type](
#             pb,
#             embodiment_id=self.embodiment_id,
#             tcp_link_id=self.tcp_link_id,
#             link_name_to_index=self.link_name_to_index,
#             joint_name_to_index=self.joint_name_to_index,
#             rest_poses=robot_arm_params['rest_poses'],
#         )
#
#         # connect a tactile sensor
#         # self.tactile_sensor = TactileSensor(
#         #     pb,
#         #     embodiment_id=self.embodiment_id,
#         #     link_name_to_index=self.link_name_to_index,
#         #     joint_name_to_index=self.joint_name_to_index,
#         #     image_size=tactile_sensor_params["image_size"],
#         #     turn_off_border=tactile_sensor_params["turn_off_border"],
#         #     sensor_type=tactile_sensor_params["type"],
#         #     sensor_core=tactile_sensor_params["core"],
#         #     sensor_dynamics=tactile_sensor_params["dynamics"],
#         #     show_tactile=tactile_sensor_params["show_tactile"],
#         #     sensor_num=1,
#         # )
#
#         self.tactile_sensors = {}
#         # 假设 tactile_sensor_params 变成了 {'left': {...}, 'right': {...}}
#         for finger in ['left', 'right']:
#             params = tactile_sensor_params[finger]
#             self.tactile_sensors[finger] = TactileSensor(
#                 pb,
#                 embodiment_id=self.embodiment_id,
#                 link_name_to_index=self.link_name_to_index,
#                 joint_name_to_index=self.joint_name_to_index,
#                 image_size=params["image_size"],
#                 turn_off_border=params["turn_off_border"],
#                 sensor_type=params["type"],
#                 sensor_core=params["core"],
#                 sensor_dynamics=params["dynamics"],
#                 show_tactile=params["show_tactile"],
#                 sensor_num=1 if finger == 'left' else 2,  # 随便定义，方便调试
#                 body_link=params.get("body_link"),  # 关键参数
#                 tip_link=params.get("tip_link"),
#             )
#
#     def load_urdf(self):
#         """
#         Load the robot arm model into pybullet
#         """
#         self.base_pos = [0, 0, 0]
#         self.base_rpy = [0, 0, 0]
#         self.base_orn = self._pb.getQuaternionFromEuler(self.base_rpy)
#         asset_name = os.path.join(
#             "embodiment_assets",
#             "combined_urdfs",
#             self.arm_type + "_" + self.tactile_sensor_type + ".urdf",
#         )
#         print("\n[DEBUG] Will load URDF file:", add_assets_path(asset_name))
#         self.embodiment_id = self._pb.loadURDF(
#             add_assets_path(asset_name), self.base_pos, self.base_orn, useFixedBase=True
#         )
#
#         # create dicts for mapping link/joint names to corresponding indices
#         self.num_joints, self.link_name_to_index, self.joint_name_to_index = self.create_link_joint_mappings(
#             self.embodiment_id)
#
#         # get the link and tcp IDs
#         self.tcp_link_id = self.link_name_to_index[self.tcp_link_name]
#
#     def reset(self, reset_tcp_pose):
#         """
#         Reset the pose of the arm and sensor
#         """
#         self.arm.reset()
#         self.tactile_sensor.reset()
#
#         # move to the initial position
#         self.arm.move_linear(reset_tcp_pose, quick_mode=True)
#
#     def full_reset(self):
#         self.load_urdf()
#         self.tactile_sensor.turn_off_collisions()
#
#     def get_tactile_observation(self):
#         return self.tactile_sensor.get_observation()


class TactileArmEmbodiment(ArmEmbodiment):
    def __init__(
        self,
        pb,
        robot_arm_params={},
        tactile_sensor_params={}
    ):
        self._pb = pb
        self.arm_type = robot_arm_params["type"]

        if "tcp_link_name" in robot_arm_params:
            self.tcp_link_name = robot_arm_params["tcp_link_name"]
        else:
            self.tcp_link_name = "ee_link"

        self.load_urdf()
        self.arm = arm_mapping[self.arm_type](
            pb,
            embodiment_id=self.embodiment_id,
            tcp_link_id=self.tcp_link_id,
            link_name_to_index=self.link_name_to_index,
            joint_name_to_index=self.joint_name_to_index,
            rest_poses=robot_arm_params['rest_poses'],
        )

        self.tactile_sensors = {}
        for finger in ['left', 'right']:
            params = tactile_sensor_params[finger]
            print("DEBUG TactileArmEmbodiment params:", params)
            self.tactile_sensors[finger] = TactileSensor(
                pb,
                embodiment_id=self.embodiment_id,
                link_name_to_index=self.link_name_to_index,
                joint_name_to_index=self.joint_name_to_index,
                image_size=params["image_size"],
                turn_off_border=params["turn_off_border"],
                sensor_type=params["type"],
                sensor_core=params["core"],
                sensor_dynamics=params["dynamics"],
                show_tactile=params["show_tactile"],
                sensor_num=0 if finger == 'left' else 1,
                body_link=params.get("body_link"),
                tip_link=params.get("tip_link"),
            )


    def reset(self, reset_tcp_pose):
        self.arm.reset()
        for sensor in self.tactile_sensors.values():
            sensor.reset()
        self.arm.move_linear(reset_tcp_pose, quick_mode=True)

    def full_reset(self):
        self.load_urdf()
        for sensor in self.tactile_sensors.values():
            sensor.turn_off_collisions()

    def get_tactile_observation(self):
        return {finger: sensor.get_observation() for finger, sensor in self.tactile_sensors.items()}

class VisualArmEmbodiment(ArmEmbodiment):
    def __init__(
        self,
        pb,
        robot_arm_params={},
        visual_sensor_params={}
    ):

        self._pb = pb
        self.arm_type = robot_arm_params["type"]

        if "tcp_link_name" in robot_arm_params:
            self.tcp_link_name = robot_arm_params["tcp_link_name"]
        else:
            self.tcp_link_name = "ee_link"

        # load the urdf file
        self.load_urdf()

        # instantiate a robot arm
        self.arm = arm_mapping[self.arm_type](
            pb,
            embodiment_id=self.embodiment_id,
            tcp_link_id=self.tcp_link_id,
            link_name_to_index=self.link_name_to_index,
            joint_name_to_index=self.joint_name_to_index,
            rest_poses=robot_arm_params['rest_poses'],
        )

        # connect a static vision sensor
        self.vision_sensor = VisionSensor(
            pb,
            sensor_num=1,
            **visual_sensor_params
        )

    def full_reset(self):
        self.load_urdf()

    def get_visual_observation(self):
        return self.vision_sensor.get_observation()


class VisuoTactileArmEmbodiment(TactileArmEmbodiment):
    def __init__(
        self,
        pb,
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={}
    ):

        self._pb = pb
        self.arm_type = robot_arm_params["type"]
        self.tactile_sensor_type = tactile_sensor_params["type"]

        if "tcp_link_name" in robot_arm_params:
            self.tcp_link_name = robot_arm_params["tcp_link_name"]
        else:
            self.tcp_link_name = "ee_link"

        # load the urdf file
        self.load_urdf()

        # instantiate a robot arm
        self.arm = arm_mapping[self.arm_type](
            pb,
            embodiment_id=self.embodiment_id,
            tcp_link_id=self.tcp_link_id,
            link_name_to_index=self.link_name_to_index,
            joint_name_to_index=self.joint_name_to_index,
            rest_poses=robot_arm_params['rest_poses'],
        )

        # connect a tactile sensor
        self.tactile_sensor = TactileSensor(
            pb,
            embodiment_id=self.embodiment_id,
            link_name_to_index=self.link_name_to_index,
            joint_name_to_index=self.joint_name_to_index,
            image_size=tactile_sensor_params["image_size"],
            turn_off_border=tactile_sensor_params["turn_off_border"],
            sensor_type=tactile_sensor_params["type"],
            sensor_core=tactile_sensor_params["core"],
            sensor_dynamics=tactile_sensor_params["dynamics"],
            show_tactile=tactile_sensor_params["show_tactile"],
            sensor_num=1,
        )

        # connect a static vision sensor
        self.vision_sensor = VisionSensor(
            pb,
            sensor_num=1,
            **visual_sensor_params
        )

    def get_visual_observation(self):
        return self.vision_sensor.get_observation()
