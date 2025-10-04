import os
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from klampt import WorldModel
import numpy as np

ROBOT_URDF_FILE = "dexee/dexee.urdf"

# TODO: rework this to use the klampt visualizer
# (apart from the pca plot)


def plot_normals(normals: np.ndarray, points: np.ndarray):

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]  #

    norms = np.linalg.norm(normals, axis=1)
    U, V, W = normals[:, 0] / norms, normals[:, 1] / norms, normals[:, 2] / norms

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(X, Y, Z, U, V, W)

    x_lim, y_lim, z_lim = X.max() + U.max(), Y.max() + V.max(), Z.max() + W.max()
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([-y_lim, y_lim])
    ax.set_zlim([-z_lim, z_lim])
    plt.show()


def load_data():
    grasp_points = []
    joint_angles = []
    object_transforms = []
    finger_assignments = []
    for file in os.listdir("grasps"):
        with open(f"grasps/{file}") as f:
            data = json.load(f)
            grasp_points.append(data["grasp_points"])
            joint_angles.append(data["joint_angles"])
            object_transforms.append(data["object_transform"])
            finger_assignments.append(data["fingertip_assignment"])

    return (
        np.array(grasp_points),
        np.array(joint_angles),
        object_transforms,
        finger_assignments,
    )


# plot grasp points in 3d
def plot_points(points: np.ndarray):

    num_grasps = points.shape[0]
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(num_grasps):

        X, Y, Z = points[i, :, 0], points[i, :, 1], points[i, :, 2]
        ax.scatter(X, Y, Z)

    x_lim, y_lim, z_lim = points.max(), points.max(), points.max()
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([-y_lim, y_lim])
    ax.set_zlim([-z_lim, z_lim])
    plt.show()


# pca for the joint angles
def do_pca(data, num_components):

    pca = PCA(n_components=num_components)

    pca.fit(data)

    return pca.transform(data)


def project_points_into_robot_space(points, joint_angles):
    world = WorldModel()
    world.loadElement(ROBOT_URDF_FILE)
    robot = world.robot(0)

    link = "hand_base"

    transformed_grasps = []
    for i, grasp in enumerate(points):
        robot.setConfig(joint_angles[i])
        R, t = robot.link(link).getTransform()
        transformed_grasps.append(transform_grasp(grasp, R, t))

    return np.array(transformed_grasps)


def project_points_into_object_space(points, object_transforms):
    transformed_grasps = []
    for i, grasp in enumerate(points):
        R, t = np.array(object_transforms[i][0]), object_transforms[i][1]
        transformed_grasps.append(transform_grasp(grasp, object_transforms[i]))
    return np.array(transformed_grasps)


def transform_grasp(grasp_points, htm):
    transformed_points = []
    for point in grasp_points:
        """rotation = np.array([R[:3], R[3:6], R[6:]])
        translation = np.array(t)
        htm = np.array(
            [
                [rotation[0, 0], rotation[0, 1], rotation[0, 2], translation[0]],
                [rotation[1, 0], rotation[1, 1], rotation[1, 2], translation[1]],
                [rotation[2, 0], rotation[2, 1], rotation[2, 2], translation[2]],
                [0, 0, 0, 1],
            ]
        )"""
        transformed_point = htm @ np.array([point[0], point[1], point[2], 1]).transpose()
        transformed_points.append(transformed_point)
    return transformed_points


def main():

    # rename
    for i, grasp in enumerate(os.listdir("grasps")):
        os.rename(f"grasps/{grasp}", f"grasps/grasp{i}.json")

    grasp_points, joint_angles, object_transforms, _ = load_data()
    plot_points(grasp_points)  # Grasp points in world space

    """pc = do_pca(joint_angles, 2)  # TODO: exclude 0 joints?
    print(pc.shape)
    plt.scatter(pc[:, 0], pc[:, 1])
    plt.show()"""

    # grasp_points_wrt_robot = project_points_into_robot_space(grasp_points, joint_angles)
    # plot_points(grasp_points_wrt_robot)

    grasp_points_wrt_object = project_points_into_object_space(grasp_points, object_transforms)
    plot_points(grasp_points_wrt_object)


if __name__ == "__main__":
    main()
