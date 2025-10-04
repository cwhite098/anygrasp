import os
import json

import numpy as np

from klampt import WorldModel
from klampt.model.create import primitives
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Slerp
from klampt.model import ik

from plots import load_data
from src.robot_config import DexeeConfig


MESH_FILENAME = "can.stl"
GRASP_FORCE_TOLERANCE = 0.1

# TODO: THIS WHOLE THING NEEDS ANOTHER LOOK


class GraspDatasetExpander:

    def __init__(self):
        # Load dataset

        # Construct (q,p)

        # Load the mesh

        # Init my KNN thingy
        pass

    def interpolate_grasp_dataset(self, n_new_grasps):
        # Expand the dataset by interpolating between existing grasps
        pass

    def grasp_rrt_expand(self):
        # Expand the dataset by extrapolating along stable manifolds in the grasp space
        raise NotImplementedError

    def _get_grasp_points_on_mesh(self, grasp_points, object_transform):
        # Given a set of grasp points and an object transform, find the nearest points on the mesh
        pass

    def _find_midpoint_transform(self, htm1, htm2) -> np.ndarray:
        # Find the midpoint transform between two homogeneous transformation matrices
        pass

    def _interpolate_rotations(self, rot1, rot2) -> np.ndarray:
        # Use slerp to interpolate between two rotation matrices
        pass

    def _solve_ik(self, grasp_points):
        # Solve the IK for a given set of grasp points
        pass


def save_grasp_data(grasp_data):
    if not os.path.exists("grasps/"):
        os.mkdir("grasps")

    n_grasps = len(os.listdir("grasps"))

    with open(f"grasps/grasp{n_grasps}.json", "w") as f:
        json.dump(grasp_data, f)


def grasp_rrt_expand():

    robot_config = DexeeConfig()

    world = WorldModel()
    world.loadElement(robot_config.urdf_path)
    object = primitives.Geometry3D()
    object.loadFile(MESH_FILENAME)
    robot = world.robot(0)

    # sample a point
    grasp_points, joint_angles, object_transforms, finger_assignments = load_data()
    num_joints = robot_config.num_joints
    num_grasps = grasp_points.shape[0]

    clustering_data = []
    for i in range(num_grasps):
        # I know the paper says to use q,p but would grasp points + object pose work better?
        grasp = np.hstack((joint_angles[i], np.array(object_transforms[i][0]).flatten()))
        clustering_data.append(grasp)

    # knn on (q,p)
    X = np.array(clustering_data)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X)
    grasp_neighbours = nbrs.kneighbors(X)[1]

    successes = 0

    # interpolate to NN
    for i, grasp_idx in enumerate(grasp_neighbours[:, 0]):
        other_grasp_idx = grasp_neighbours[i, 1]
        # midpoint_grasp = X[grasp_idx] - X[other_grasp_idx]
        midpoint_joints = (X[grasp_idx, :num_joints] + X[other_grasp_idx, :num_joints]) / 2

        grasp_object_tf = np.array(object_transforms[grasp_idx][0])
        other_grasp_object_tf = np.array(object_transforms[other_grasp_idx][0])
        # interpolate these matrices with slerp
        interp_object_rotation = None

        slerp = Slerp([0, 1], R.from_matrix([grasp_object_tf, other_grasp_object_tf]))
        interp_object_rotation = slerp(0.5)

        # get grasp point midpoints
        grasp = grasp_points[grasp_idx]

        other_grasp = grasp_points[other_grasp_idx]
        other_finger_assigment = finger_assignments[other_grasp_idx]

        grasp_midpoints = []
        for i, finger in enumerate(finger_assignments[grasp_idx]):
            other_finger_idx = other_finger_assigment.index(finger)
            point = (grasp[:, i] + other_grasp[:, other_finger_idx]) / 2
            grasp_midpoints.append(point)

        grasp_midpoints = np.array(grasp_midpoints)

        # fix contact and collision
        #   find nearest point on mesh to midpoint
        #   do solve ik with that point and midpoint of q as init
        object_mesh_points, _ = sample_mesh(MESH_FILENAME, interp_object_rotation)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(object_mesh_points[:, :3])
        mesh_point_idx = nbrs.kneighbors(grasp_midpoints)[1]
        new_grasp_points = object_mesh_points[mesh_point_idx.flatten(), :3]

        ik_objectives = []
        for i, fingertip in enumerate(finger_assignments[grasp_idx]):
            ik_objectives.append(
                ik.objective(
                    robot.link(fingertip),
                    local=[0.0, -0.012304481602826016, 5.551115123125783e-17],
                    world=new_grasp_points[i],
                )
            )

        def feasible():
            collision_detected = False
            for link in robot.links:
                if object.collides(link.geometry()):
                    collision_detected = True
                    continue
            return not robot.selfCollides() and not collision_detected

        robot.setConfig(midpoint_joints)

        success = ik.solve_global(
            ik_objectives,
            numRestarts=1000,
            iters=300,
            feasibilityCheck=feasible,
            startRandom=True,
        )
        print(success)
        if success:
            successes += 1
            grasp_config = robot.getConfig()

            # we need to transform into robot space here too since the ik will have thrown it off

        # maybe we want force optim too ?
    print(successes)
    # would be nice to visualise each grasp and the interped one
