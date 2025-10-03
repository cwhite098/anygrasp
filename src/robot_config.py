from dataclasses import dataclass

from klampt import WorldModel

IDENTITY_ROTATION_MAT_FLAT = [1, 0, 0, 0, 1, 0, 0, 0, 1]
IDENTITY_TRANSLATION = [0, 0, 0]


@dataclass
class DexeeConfig:

    name: str = "dexee"
    num_fingers: int = 3
    urdf_path: str = "dexee/dexee.urdf"
    base_link: str = "hand_base"  # link after the virtual arm

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
        assert t == IDENTITY_TRANSLATION
        assert R == IDENTITY_ROTATION_MAT_FLAT

    def _get_num_joints(self, robot):
        # TODO: figure out what to do about fixed joints not being collapsed
        self.num_joints: int = len(robot.getConfig())

    def _get_fingertip_links(self, robot):
        # This will work for dexee and hande but maybe not for other hands
        self.fingertip_links: list[str] = [
            link.name for link in robot.links if "tip" in link.name
        ]
        assert len(self.fingertip_links) == self.num_fingers


def main():
    robot_config = DexeeConfig()


if __name__ == "__main__":
    main()
