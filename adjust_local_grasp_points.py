from klampt.model import ik
from klampt import WorldModel
from klampt import vis
from itertools import permutations
import numpy as np

from robot_config import DexeeConfig
from vis_grasp import invert_htm


def main():
    from klampt.io import resource

    world = WorldModel()

    ROBOT_URDF_FILE = "dexee/dexee.urdf"

    world.loadElement(ROBOT_URDF_FILE)

    robot = world.robot(0)

    link = robot.link("F2_tip")

    (save, value) = resource.edit(
        "Local point", [0, 0, 0], type="Point", world=world, frame=link
    )
    if save:
        localpt = value
    (save, value) = resource.edit("World point", [0, 0, 0], type="Point", frame=None)
    if save:
        worldpt = value


if __name__ == "__main__":
    main()
