import os
import json

import numpy as np
from sklearn.decomposition import PCA

from klampt import WorldModel
from klampt import vis
import numpy as np

from vis_grasp import ORIGIN
from src.robot_config import DexeeConfig


def load_data():
    grasp_points = []
    joint_angles = []
    object_transforms = []
    finger_assignments = []
    for file in os.listdir("grasps"):
        with open(f"grasps/{file}") as f:
            data = json.load(f)
            grasp_points.append(data["contact_points"])
            joint_angles.append(data["grasp_config"])
            object_transforms.append(data["object_htm"])
            finger_assignments.append(data["fingertip_assigment"])

    return (
        np.array(grasp_points),
        np.array(joint_angles),
        np.array(object_transforms),
        finger_assignments,
    )


# plot grasp points in 3d
def plot_points(points: np.ndarray, robot):

    vis.add("Origin", ORIGIN)
    for i, grasp in enumerate(points):
        for j, point in enumerate(grasp):
            vis.add(f"grasp {i} point {j}", point)

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

    # Clean up the grasp data directory
    for i, grasp in enumerate(os.listdir("grasps")):
        os.rename(f"grasps/{grasp}", f"grasps/grasp{i}.json")

    grasp_points, joint_angles, object_transforms, _ = load_data()
    world = WorldModel()
    world.loadElement(DexeeConfig().urdf_path)
    robot = world.robot(0)

    plot_points(grasp_points, robot)  # Grasp points in robot/world space


if __name__ == "__main__":
    main()
