import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

urdf_path = r"H:/tactile-gym-3-bowen/tg3/simulator/assets/robot_arm_assets/ur5/urdfs/ur5.urdf"
# urdf_path = r"H:/tactile-gym-3-bowen/tg3/simulator/assets/embodiment_assets/combined_urdfs/ur5_standard_tactip.urdf"
# urdf_path = r"H:/tactile-gym-3-bowen/tg3/simulator/assets/embodiment_assets/ur5_standard_tactip.urdf"
# urdf_path = r"H:/tactile-gym-3-bowen/tg3/simulator/robots/arms/robotiq_arg85/robots/robotiq_arg85_description.URDF"
gripper = p.loadURDF(urdf_path, useFixedBase=True)

while True:
    p.stepSimulation()
