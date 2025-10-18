import os
import json

import numpy as np
from stl import mesh
from klampt import WorldModel, vis
from klampt.model.create import primitives
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Slerp
from klampt.model import ik
from klampt.math import se3

from plots import load_data
from src.robot_config import DexeeConfig
from src.grasp_generator import Grasp


class GraspDatasetExpander:

    def __init__(self, mesh_filename: str, grasp_dir: str, save_dir: str):
        self.mesh_path = mesh_filename
        self.save_dir = save_dir

        # Load dataset
        self.grasp_points, self.joint_angles, self.object_htms, self.finger_assignments = load_data(grasp_dir)

        num_grasps = self.grasp_points.shape[0]

        # Construct (q,p)
        self.grasp_embeddings = []
        for i in range(num_grasps):
            # how to we want to represent the object pose? pos+quat
            grasp = np.hstack((self.joint_angles[i], np.array(self.object_htms[i][0]).flatten()))
            self.grasp_embeddings.append(grasp)

        # Load the robot + object
        self.robot_config = DexeeConfig()
        self.world = WorldModel()
        self.world.loadElement(self.robot_config.urdf_path)
        self.object = primitives.Geometry3D()
        self.object.loadFile(mesh_filename)
        self.robot = self.world.robot(0)

        self.object_ghost = primitives.Geometry3D()
        self.object_ghost.loadFile(mesh_filename)
        self.world.loadElement(self.robot_config.urdf_path)
        self.robot_ghost = self.world.robot(1)

    def interpolate_grasp_dataset(self, n_new_grasps, visualise: bool = False):
        # Expand the dataset by interpolating between existing grasps

        grasp_knn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(self.grasp_embeddings)
        grasp_neighbours = grasp_knn.kneighbors(self.grasp_embeddings)[1]

        # interpolate to NN
        grasps_generated = 0
        while grasps_generated < n_new_grasps:

            # TODO: integrate with the new grasp dataset

            grasp_idx = np.random.randint(0, self.grasp_points.shape[0])
            other_grasp_idx = grasp_neighbours[grasp_idx, 1]

            # JOINT MIDPOINT
            midpoint_joints = (self.joint_angles[grasp_idx] + self.joint_angles[other_grasp_idx]) / 2

            # OBJECT HTM MIDPOINT
            interp_htm = self._find_midpoint_transform(
                np.array(self.object_htms[grasp_idx]), np.array(self.object_htms[other_grasp_idx])
            )

            # GRASP POINT MIDPOINTS
            new_grasp_points = self._get_grasp_points_on_mesh(
                self.grasp_points[grasp_idx],
                self.finger_assignments[grasp_idx],
                self.grasp_points[other_grasp_idx],
                self.finger_assignments[other_grasp_idx],
                interp_htm,
            )

            self.object.setCurrentTransform(se3.from_homogeneous(interp_htm)[0], se3.from_homogeneous(interp_htm)[1])
            self.robot.setConfig(np.concat(([0, 0, 0, 0, 0, 0], self.joint_angles[grasp_idx])).tolist())
            valid_grasp = self._solve_ik(grasp_idx, new_grasp_points, midpoint_joints)

            if valid_grasp:
                grasps_generated += 1

                robot_config = self.robot.getConfig()[6:]  # exclude virtual arm

                grasp = Grasp(
                    contact_points=new_grasp_points.tolist(),
                    grasp_config=robot_config,
                    object_htm=interp_htm.tolist(),
                    fingertip_assigment=self.finger_assignments[grasp_idx],
                )
                self.save_grasp_data(self.save_dir, grasp._asdict())
                if visualise:
                    self._visualise_interpolation(grasp_idx)

    def grasp_rrt_expand(self):
        raise NotImplementedError

    def _get_grasp_points_on_mesh(self, grasp_point1, fingers1, grasp_point2, fingers2, object_htm):
        # Given a set of grasp points and an object transform, find the nearest points on the mesh

        new_grasp_points = []
        for i, point in enumerate(grasp_point1):
            finger = fingers1[i]
            # Get index of same finger in other grasp
            other_idx = fingers2.index(finger)

            midpoint = (point + grasp_point2[other_idx]) / 2

            # Find nearest point on mesh to midpoint
            object_mesh_points, _ = self._sample_mesh(object_htm)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(object_mesh_points[:, :3])
            mesh_point_idx = nbrs.kneighbors(midpoint.reshape(1, -1))[1]
            new_point1 = object_mesh_points[mesh_point_idx.flatten(), :3]

            new_grasp_points.append(new_point1.flatten())

        return np.array(new_grasp_points)

    def _find_midpoint_transform(self, htm1, htm2) -> np.ndarray:
        # Find the midpoint transform between two homogeneous transformation matrices
        slerp = Slerp([0, 1], R.from_matrix([htm1[:3, :3], htm2[:3, :3]]))
        interp_object_rotation = slerp(0.5)

        interp_object_translation = (htm1[:3, 3] + htm2[:3, 3]) / 2

        interp_object_htm = np.eye(4)
        interp_object_htm[:3, :3] = interp_object_rotation.as_matrix()
        interp_object_htm[:3, 3] = interp_object_translation
        return interp_object_htm

    def _solve_ik(self, grasp_idx, new_grasp_points, midpoint_joints) -> bool:
        # Solve the IK for a given set of grasp points
        ik_objectives = []

        for i, fingertip in enumerate(self.finger_assignments[grasp_idx]):
            ik_objectives.append(
                ik.objective(
                    self.robot.link(fingertip),
                    local=self.robot_config.fingertip_grasp_offset,
                    world=new_grasp_points[i],
                )
            )

        def feasible():
            for link in self.robot.links:
                if self.object.collides(link.geometry()):
                    return False
            return not self.robot.selfCollides()

        self.robot.setConfig(np.concat(([0, 0, 0, 0, 0, 0], midpoint_joints)).tolist())

        success = ik.solve_global(
            ik_objectives,
            numRestarts=1000,
            iters=300,
            feasibilityCheck=feasible,
            activeDofs=range(6, self.robot_config.num_joints),
        )
        return success

    def _sample_mesh(self, object_htm) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a mesh file (.stl) and a scipy rotation, provide the points and surface normals
        of the mesh in world space.
        """
        object_mesh = mesh.Mesh.from_file(self.mesh_path)
        object_mesh.transform(object_htm)

        # Include the all the vertices
        points = np.vstack(
            [
                object_mesh.points[:, :3],
                object_mesh.points[:, 3:6],
                object_mesh.points[:, 6:9],
            ]
        )
        normals = np.vstack([object_mesh.normals] * 3)

        return points, normals

    def _visualise_interpolation(self, grasp_idx):
        self.object_ghost.setCurrentTransform(
            se3.from_homogeneous(self.object_htms[grasp_idx])[0],
            se3.from_homogeneous(self.object_htms[grasp_idx])[1],
        )
        self.robot_ghost.setConfig(np.concat(([0, 0, 0, 0, 0, 0], self.joint_angles[grasp_idx])).tolist())
        vis.clear()
        vis.add("object", self.object)
        vis.add("object_ghost", self.object_ghost, color=[0, 1, 0, 0.5])
        vis.add("robot", self.robot)
        vis.add("robot_ghost", self.robot_ghost, color=[0, 1, 0, 0.5])
        vis.add("origin", np.eye(4))
        vis.autoFitCamera()
        vis.run()

    @staticmethod
    def save_grasp_data(grasp_dir, grasp_data):
        if not os.path.exists(grasp_dir):
            os.mkdir(grasp_dir)

        n_grasps = len(os.listdir(grasp_dir))

        with open(f"{grasp_dir}/int_grasp{n_grasps}.json", "w") as f:
            json.dump(grasp_data, f)
