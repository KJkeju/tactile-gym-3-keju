from tg3.simulator.robots.arms.mg400.mg400 import MG400
from tg3.simulator.robots.arms.ur5.ur5 import UR5
from tg3.simulator.robots.arms.franka_panda.franka_panda import FrankaPanda
from tg3.simulator.robots.arms.kuka_iiwa.kuka_iiwa import KukaIiwa
from tg3.simulator.robots.arms.cr3.cr3 import CR3

arm_mapping = {
    "ur5": UR5,
    "franka_panda": FrankaPanda,
    "kuka_iiwa": KukaIiwa,
    "mg400": MG400,
    "cr3": CR3,
}