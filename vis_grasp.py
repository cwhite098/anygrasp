from klampt import WorldModel, math
import numpy as np
from klampt.model.create import primitives
from scipy.spatial.transform import Rotation as R
from klampt import vis
import json
import os

ORIGIN = [[1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]]


def invert_htm(htm: np.ndarray):
    rot_trans = htm[:3, :3].transpose()

    translation = -rot_trans @ htm[:3, 3]
    inverse = np.array(
        [
            [rot_trans[0, 0], rot_trans[0, 1], rot_trans[0, 2], translation[0]],
            [rot_trans[1, 0], rot_trans[1, 1], rot_trans[1, 2], translation[1]],
            [rot_trans[2, 0], rot_trans[2, 1], rot_trans[2, 2], translation[2]],
            [0, 0, 0, 1],
        ]
    )
    return inverse


def main():

    ROBOT_URDF_FILE = "dexee/dexee.urdf"

    object_file = "can.stl"

    for grasp_file in os.listdir("grasps"):
        # for grasp_file in ["grasp21.json", "grasp22.json", "grasp23.json"]:
        with open(f"grasps/{grasp_file}") as f:
            grasp_data = json.load(f)

        world = WorldModel()
        world.loadElement(ROBOT_URDF_FILE)
        object = primitives.Geometry3D()
        object.loadFile(object_file)
        robot = world.robot(0)

        # Load object wrt world
        object_htm_wrt_world = np.array(grasp_data["object_htm_wrt_world"])
        object_rotation_wrt_world = [i for i in object_htm_wrt_world[:3, :3].flatten()]
        object_translation_wrt_world = [i for i in object_htm_wrt_world[:3, 3]]
        print("###### OBJECT WRT WORLD")
        print(object_htm_wrt_world)
        print("###### INV OBJECT WRT WORLD")
        print(invert_htm(object_htm_wrt_world))

        # Load object wrt robot
        object_htm_wrt_robot = np.array(grasp_data["object_htm_wrt_robot"])
        object_rotation_wrt_robot = [i for i in object_htm_wrt_robot[:3, :3].flatten()]
        object_translation_wrt_robot = [i for i in object_htm_wrt_robot[:3, 3]]
        print("###### OBJECT WRT ROBOT")
        print(object_htm_wrt_robot)
        print("###### INV OBJECT WRT ROBOT")
        print(invert_htm(object_htm_wrt_robot))

        # Load robot wrt world
        robot_htm_wrt_world = np.array(grasp_data["robot_htm_wrt_world"])
        robot_rotation_wrt_world = [i for i in robot_htm_wrt_world[:3, :3].flatten()]
        robot_translation_wrt_world = [i for i in robot_htm_wrt_world[:3, 3]]
        print("###### ROBOT WRT WORLD")
        print(robot_htm_wrt_world)
        print("###### INV ROBOT WRT WORLD")
        print(invert_htm(robot_htm_wrt_world))

        # Check robot frame matches with result of virtual arm
        robot.setConfig(grasp_data["joint_angles"])
        robot_grasp_transform = robot.link("hand_base").getTransform()
        assert robot_rotation_wrt_world == robot_grasp_transform[0]
        assert robot_translation_wrt_world == robot_grasp_transform[1]
        print("THIS IS IT: ", robot.link("hand_base").getLocalPosition([0, 0, 0]))
        translation = robot.link("hand_base").getLocalPosition([0, 0, 0])

        print(invert_htm(robot_htm_wrt_world) @ object_htm_wrt_world)
        print(object_htm_wrt_robot)
        assert np.allclose(
            (invert_htm(robot_htm_wrt_world) @ object_htm_wrt_world).flatten(),
            object_htm_wrt_robot.flatten(),
        )

        grasp_data["joint_angles"][:6] = [0] * 6
        robot.setConfig(grasp_data["joint_angles"])
        print(robot.link("hand_base").getTransform())
        assert robot.link("hand_base").getTransform() == (
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        )

        # try to transform robot with inverse robot_htm_wrt_world to see where hand_base ends up

        # object_htm_wrt_robot = invert_htm(robot_htm_wrt_world) @ object_htm_wrt_world
        transform = np.eye(4) @ object_htm_wrt_robot

        # translation = robot_htm_wrt_world @ np.array(robot_translation_wrt_world)
        trans = [i for i in transform[:3, 3]]
        rotation = [i for i in transform[:3, :3].flatten()]

        object.setCurrentTransform(rotation, trans)

        vis.kill()
        vis.add("robot", robot)
        vis.add("object", object)
        vis.add("origin", ORIGIN)
        vis.run()


if __name__ == "__main__":
    main()
