from itertools import permutations
import time
from multiprocessing import Lock

from scipy.spatial.transform import Rotation as R
import numpy as np
from stl import mesh
from klampt import vis, PointCloud, WorldModel
from klampt.model import ik
from klampt.model.create import primitives
from klampt.math import se3
from klampt.model.contact import ContactPoint, force_closure

from vis_grasp import ORIGIN
from src.anygrasp.robot_config import RobotConfig
from src.anygrasp.dataset import GraspDataset, Grasp


def klampt_transform_to_htm(rot: list[float], t: list[float]) -> np.ndarray:
    """
    Transform a klampt rotation and translation into a homogeneous transformation matrix.
    Klampt is column major!!
    """
    rotation = np.array([rot[:3], rot[3:6], rot[6:]])
    translation = np.array(t)
    htm = np.array(
        [
            [rotation[0, 0], rotation[1, 0], rotation[2, 0], translation[0]],
            [rotation[0, 1], rotation[1, 1], rotation[2, 1], translation[1]],
            [rotation[0, 2], rotation[1, 2], rotation[2, 2], translation[2]],
            [0, 0, 0, 1],
        ]
    )
    return htm


class GraspGenerator:
    def __init__(
        self,
        robot_config: RobotConfig,
        stl_path: str,
        num_grasp_points: int,
        grasp_force_tolerance: float = 0.1,
        visualise: bool = False,
        save_dir: str | None = None,
    ):
        self.robot_config: RobotConfig = robot_config
        self.stl_path = stl_path
        self.grasp_force_tolerance = grasp_force_tolerance
        self.num_grasp_points = num_grasp_points
        self.visualise = visualise
        self.save_dir = save_dir

        self.world = WorldModel()

        self.object = primitives.Geometry3D()
        self.object.loadFile(self.stl_path)

        self.world.loadElement(robot_config.urdf_path)
        self.robot = self.world.robot(0)

        if self.visualise:
            vis.clear()
            vis.add("object", self.object)
            vis.add("origin", ORIGIN)
            vis.autoFitCamera(rotate=False, scale=0.7)
            vis.add("robot", self.robot)
            vis.show()

    def analyse_grasp_stability(self):
        # Gonna run with this for now cos simple
        return force_closure(self.contact_points)

    def load_object(self, rotation) -> tuple[np.ndarray, np.ndarray]:
        points, normals = self._sample_mesh(rotation)

        for i, n in enumerate(normals):
            n = n / np.linalg.norm(n)
            normals[i] = n

        object_rotation_wrt_world = self.object_rotation.as_matrix()
        # transpose because klampt is column major
        rot, t = se3.from_rotation(object_rotation_wrt_world.transpose())

        self.object.setCurrentTransform(rot, t)
        self.object_com = points.mean(axis=0).tolist()

        self.object_pointcloud = PointCloud()
        self.object_pointcloud.setPoints(points)
        return points, normals

    def solve_ik(self, grasp_points_wrt_world):

        for link_perm in permutations(self.robot_config.fingertip_links, self.num_grasp_points):
            ik_objectives = []
            for i, fingertip in enumerate(link_perm):
                ik_objectives.append(
                    ik.objective(
                        self.robot.link(fingertip),
                        local=self.robot_config.fingertip_grasp_offset,
                        world=grasp_points_wrt_world[i],
                    )
                )

            def feasible():
                for link in self.robot.links:
                    if self.object.collides(link.geometry()):
                        return False
                return not self.robot.selfCollides()

            # If we are executing rough grasps with a policy, can we use solve_nearby?
            success = ik.solve_global(
                ik_objectives,
                numRestarts=200,
                iters=200,
                feasibilityCheck=feasible,
                startRandom=True,
            )
            if success:
                self.grasp_config = self.robot.getConfig()
                return True, link_perm
        return False, link_perm

    def transform_into_robot_space(self) -> np.ndarray:
        """
        Transform the grasp points and object into the robot space.
        """
        # Transform the grasp and object into robot space
        object_rotation = self.object_rotation.as_matrix()
        object_htm_wrt_world = klampt_transform_to_htm(object_rotation.flatten(), [0, 0, 0])

        rot, t = self.robot.link(self.robot_config.base_link).getTransform()
        robot_htm_wrt_world = klampt_transform_to_htm(rot, t)
        object_htm_wrt_robot = np.linalg.inv(robot_htm_wrt_world) @ object_htm_wrt_world

        self.grasp_config[:6] = [0, 0, 0, 0, 0, 0]  # reset the virtual arm (hand base -> origin)
        self.robot.setConfig(self.grasp_config)
        self.robot.link(self.robot_config.base_link).setTransform(ORIGIN[0], ORIGIN[1])

        # Transform the contact points and object pointcloud into robot space
        for i, point in enumerate(self.contact_points):
            point.transform(se3.from_homogeneous(np.linalg.inv(object_htm_wrt_world)))
            point.transform(se3.from_homogeneous(object_htm_wrt_robot))

        rot, t = se3.from_homogeneous(np.linalg.inv(object_htm_wrt_world))
        self.object_pointcloud.transform(rot, t)
        rot, t = se3.from_homogeneous(object_htm_wrt_robot)
        self.object_pointcloud.transform(rot, t)

        rot, t = se3.from_homogeneous(object_htm_wrt_robot)
        self.object.setCurrentTransform(rot, t)

        return object_htm_wrt_robot

    def generate_grasp(self, visualise: bool = False, lock = None) -> Grasp:
        self.object_rotation = R.random()

        points, normals = self.load_object(self.object_rotation)

        # Generate the grasp
        is_stable = False
        is_valid = False
        print("=================")
        while not is_valid:
            grasp_point_idx = np.random.randint(0, points.shape[0], self.num_grasp_points)
            self.contact_points = [
                ContactPoint(p, n, kFriction=0.1) for p, n in zip(points[grasp_point_idx], normals[grasp_point_idx])
            ]
            is_stable = self.analyse_grasp_stability()
            if is_stable:
                is_valid, link_perm = self.solve_ik(points[grasp_point_idx])

        print("Found valid grasp")
        object_htm_wrt_robot = self.transform_into_robot_space()
        contact_points_array = np.array([cp.x for cp in self.contact_points])

        # Remove the fixed joints from the grasp config
        excluded_dofs = []
        for i in range(self.robot_config.num_joints):
            if self.robot.getJointType(i) == "weld":
                excluded_dofs.append(i)
        joint_angles = np.delete(self.grasp_config, excluded_dofs).tolist()[6:] # exclude the virtual arm

        grasp = Grasp(
            robot_name=self.robot_config.name,
            object_name=self.stl_path,
            contact_points=contact_points_array.tolist(),
            joint_angles=joint_angles,
            object_htm=object_htm_wrt_robot.tolist(),
            fingertip_assignment=link_perm,
        )

        if self.visualise:
            vis.clear()
            for i, point in enumerate(self.contact_points):
                vis.add(f"contact point {i}", point)
            vis.add("object pointcloud", self.object_pointcloud)
            vis.add("object", self.object)
            vis.add("robot", self.robot)
            vis.add("origin", ORIGIN)
            time.sleep(1)
            vis.show()
            vis.clear()
            vis.add("object", self.object)
            vis.add("robot", self.robot)

        if lock is not None:
            lock.acquire()
        if self.save_dir is not None:
            GraspDataset.save_grasp(grasp, self.save_dir)
        if lock is not None:
            lock.release()

        return grasp

    def _sample_mesh(self, object_rotation) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a mesh file (.stl) and a scipy rotation, provide the points and surface normals
        of the mesh in world space.
        """
        object_rotation = object_rotation.as_rotvec()

        object_mesh = mesh.Mesh.from_file(self.stl_path)
        object_mesh.rotate(axis=object_rotation, theta=float(np.linalg.norm(object_rotation)))
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
