from klampt import WorldModel
from klampt.io import resource

from src.robot_config import ShadowHandConfig


def main():

    world = WorldModel()

    robot_config = ShadowHandConfig()

    world.loadElement(robot_config.urdf_path)

    robot = world.robot(0)

    link = robot.link("lftip")

    (save, value) = resource.edit("Local point", [0, 0, 0], type="Point", world=world, frame=link)
    if save:
        localpt = value
    (save, value) = resource.edit("World point", [0, 0, 0], type="Point", frame=None)
    if save:
        worldpt = value


if __name__ == "__main__":
    main()
