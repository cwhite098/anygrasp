import os
import time

from sklearn.decomposition import PCA
import numpy as np

from klampt import WorldModel
from klampt import vis
from klampt.math import se3
from klampt.model.create import primitives

from src.anygrasp.robot_config import DexeeConfig, ORIGIN, AllegroConfig
from src.anygrasp.dataset import GraspDataset, Grasp



# plot grasp points in 3d
def plot_points(grasps: list[Grasp], robot):

    vis.add("Origin", ORIGIN)
    for i, grasp in enumerate(grasps):
        for j, point in enumerate(grasp.contact_points):
            vis.add(f"grasp {i} point {j}", point, hide_label=True, color=[0, 1, 0, 1])

    if robot is not None:
        vis.add("robot", robot, color=[1, 0.8, 0.8, 0.5])
    vis.autoFitCamera()
    vis.run()
    vis.clear()
    vis.kill()


# pca for the joint angles
def do_pca(data, num_components):

    pca = PCA(n_components=num_components)

    pca.fit(data)

    return pca.transform(data)


def main():
    #grasps = GraspDataset.load_data()
    grasp_dataset = GraspDataset("grasps_allegro_pads_interp")
    world = WorldModel()
    world.loadElement(AllegroConfig().urdf_path)
    robot = world.robot(0)

    object_ghost = primitives.Geometry3D()
    object_ghost.loadFile("objects/can.stl")

    plot_points(grasp_dataset.grasps, robot)  # Grasp points in robot/world space
    plot_points([grasp_dataset.sample(GraspDataset.SamepleMode.SPECIFY, idx=0)], robot)

    for i in range(len(grasp_dataset.grasps)):
        grasp = grasp_dataset.sample(GraspDataset.SamepleMode.SPECIFY, idx=i)
        object_ghost.setCurrentTransform(
            se3.from_homogeneous(grasp.object_htm)[0],
            se3.from_homogeneous(grasp.object_htm)[1],
        )

        print(f"Grasp {i} joint angles: {grasp.joint_angles}")
        joints = np.concatenate(([0, 0, 0, 0, 0, 0], grasp.joint_angles)).tolist()
        for i in range(len(robot.getConfig())):
            if robot.getJointType(i) == "weld":
                joints = np.insert(joints, i, 0)
        robot.setConfig(joints)
        vis.add(f"grasp_{i}_robot", robot, color=[1, 0.8, 0.8, 0.5])
        vis.add(f"grasp_{i}_object", object_ghost, color=[0.8, 0.8, 1, 0.5])
        vis.autoFitCamera()
        vis.run()
        vis.clear()


if __name__ == "__main__":
    main()
