from dataclasses import dataclass, field

from klampt import WorldModel

from vis_grasp import ORIGIN


@dataclass
class RobotConfig:
    name: str
    num_fingers: int
    urdf_path: str
    base_link: str  # link after the virtual arm

    fingertip_grasp_offset: list[float]

    def __post_init__(self):
        world = WorldModel()
        world.loadElement(self.urdf_path)
        robot = world.robot(0)

        self._verify_base_link_transform(robot)
        self._get_num_joints(robot)
        self._get_fingertip_links(robot)

    def _verify_base_link_transform(self, robot):
        """
        We want to ensure that the base link transform wrt is the identity.
        This will save pain later when we try and load grasps with the grasp points wrt the world frame.
        """
        world = WorldModel()
        world.loadElement(self.urdf_path)
        robot = world.robot(0)

        base_link = robot.link(self.base_link)
        R, t = base_link.getTransform()
        assert t == ORIGIN[1]
        assert R == ORIGIN[0]

    def _get_num_joints(self, robot):
        # TODO: figure out what to do about fixed joints not being collapsed
        self.num_joints: int = len(robot.getConfig())

    def _get_fingertip_links(self, robot):
        # This will work for dexee and hande but maybe not for other hands
        self.fingertip_links: list[str] = [link.name for link in robot.links if "tip" in link.name]
        assert len(self.fingertip_links) == self.num_fingers


@dataclass
class DexeeConfig(RobotConfig):

    name: str = "dexee"
    num_fingers: int = 3
    urdf_path: str = "dexee/dexee.urdf"
    base_link: str = "hand_base"  # link after the virtual arm

    fingertip_grasp_offset: list[float] = field(
        default_factory=lambda: [0.0, -0.011804481602826016, 5.551115123125783e-17]
    )


@dataclass
class ShadowHandConfig(RobotConfig):

    name: str = "shadow_hand"
    num_fingers: int = 5
    urdf_path: str = "shadow_hand/shadow_hand_right.urdf"
    base_link: str = "forearm"  # link after the virtual arm

    fingertip_grasp_offset: list[float] = field(default_factory=lambda: [0.0, -0.007938491873549683, 0.0])


def main():
    robot_config = DexeeConfig()


if __name__ == "__main__":
    main()
