import os

from sklearn.decomposition import PCA

from klampt import WorldModel
from klampt import vis

from src.anygrasp.robot_config import DexeeConfig, ORIGIN
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
    grasps = GraspDataset.load_data()
    grasp_dataset = GraspDataset("grasps")
    world = WorldModel()
    world.loadElement(DexeeConfig().urdf_path)
    robot = world.robot(0)

    plot_points(grasps, robot)  # Grasp points in robot/world space
    plot_points([grasp_dataset.sample(GraspDataset.SamepleMode.SPECIFY, idx=0)], robot)


if __name__ == "__main__":
    main()
