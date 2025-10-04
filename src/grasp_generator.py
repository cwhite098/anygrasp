from itertools import permutations
import time

from scipy.spatial.transform import Rotation as R
import numpy as np
from stl import mesh
from klampt import vis, PointCloud, WorldModel
from klampt.model import ik
from klampt.model.create import primitives
from klampt.math import se3
from klampt.model.contact import ContactPoint, force_closure

from vis_grasp import ORIGIN


class GraspGenerator:
    def __init__(
        self,
        robot_config,
        stl_path,
        num_grasp_points,
        grasp_force_tolerance=0.1,
        visualise=False,
    ):
        self.robot_config = robot_config
        self.stl_path = stl_path
        self.grasp_force_tolerance = grasp_force_tolerance
        self.num_grasp_points = num_grasp_points
        self.visualise = visualise

        self.world = WorldModel()

        self.object = primitives.Geometry3D()
        self.object.loadFile(self.stl_path)

        if self.visualise:
            vis.clear()
            vis.add("object", self.object)
            vis.add("origin", ORIGIN)
            vis.autoFitCamera(rotate=False, scale=0.7)
            vis.show()

        self.world.loadElement(robot_config.urdf_path)
        self.robot = self.world.robot(0)

        if self.visualise:
            vis.add("robot", self.robot)

    def analyse_grasp_stability(self):
        # Gonna run with this for now cos simple
        return force_closure(self.contact_points)

    def load_object(self, rotation):
        points, normals = self._sample_mesh(rotation)

        norm_normals = []
        for n in normals:
            n = n / np.linalg.norm(n)
            norm_normals.append(n)
        normals = np.array(norm_normals)
        print(normals.shape)

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
                        local=[0.0, -0.011804481602826016, 5.551115123125783e-17],
                        world=grasp_points_wrt_world[i],
                    )
                )

            def feasible():
                # zcollision_detected = False
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
                return True
        return False

    def generate_grasp(self, visualise=False):
        self.object_rotation = R.random()

        points, normals = self.load_object(self.object_rotation)

        # Generate the grasp
        is_stable = False
        is_valid = False
        print("=================")
        while not is_valid:
            grasp_point_idx = np.random.randint(0, points.shape[0], self.robot_config.num_fingers)
            self.contact_points = [
                ContactPoint(p, n, kFriction=0.1) for p, n in zip(points[grasp_point_idx], normals[grasp_point_idx])
            ]

            is_stable = self.analyse_grasp_stability()
            print("is stable:", is_stable)

            if is_stable:
                is_valid = self.solve_ik(points[grasp_point_idx])

        # Now we want to transform this into the robot frame, the whole lot and visualise it as we go
        object_rotation = self.object_rotation.as_matrix()
        object_htm_wrt_world = np.array(
            [
                [
                    object_rotation[0, 0],
                    object_rotation[1, 0],
                    object_rotation[2, 0],
                    0,
                ],
                [
                    object_rotation[0, 1],
                    object_rotation[1, 1],
                    object_rotation[2, 1],
                    0,
                ],
                [
                    object_rotation[0, 2],
                    object_rotation[1, 2],
                    object_rotation[2, 2],
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )

        rot, t = self.robot.link(self.robot_config.base_link).getTransform()
        rotation = np.array([rot[:3], rot[3:6], rot[6:]])
        translation = np.array(t)
        robot_htm_wrt_world = np.array(
            [
                [rotation[0, 0], rotation[1, 0], rotation[2, 0], translation[0]],
                [rotation[0, 1], rotation[1, 1], rotation[2, 1], translation[1]],
                [rotation[0, 2], rotation[1, 2], rotation[2, 2], translation[2]],
                [0, 0, 0, 1],
            ]
        )
        object_htm_wrt_robot = np.linalg.inv(robot_htm_wrt_world) @ object_htm_wrt_world

        self.grasp_config[:6] = [0, 0, 0, 0, 0, 0]  # reset the virtual arm to identity
        self.robot.setConfig(self.grasp_config)
        self.robot.link("hand_base").setTransform([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0])

        for i, point in enumerate(self.contact_points):
            rot, t = se3.from_homogeneous(np.linalg.inv(object_htm_wrt_world))
            point.transform((rot, t))
            rot, t = se3.from_homogeneous(object_htm_wrt_robot)
            point.transform((rot, t))

        transformed_pc = []
        for point in self.object_pointcloud.getPoints():
            p = np.array([point[0], point[1], point[2], 1])
            new_point = object_htm_wrt_robot @ (np.linalg.inv(object_htm_wrt_world) @ p)
            transformed_pc.append(new_point[:3])
        transformed_pc = np.array(transformed_pc)
        self.object_pointcloud.setPoints(transformed_pc)

        # self.object.setCurrentTransform(ORIGIN[0], ORIGIN[1])
        rot, t = se3.from_homogeneous(object_htm_wrt_robot)
        self.object.setCurrentTransform(rot, t)

        if self.visualise:
            vis.clear()
            for i, point in enumerate(self.contact_points):
                vis.add(f"contact point {i}", point)
            vis.add("object pointcloud", self.object_pointcloud)
            vis.add("object", self.object)
            vis.add("robot", self.robot)
            vis.add("origin", ORIGIN)
            time.sleep(1)
            vis.clear()
            vis.add("object", self.object)
            vis.add("robot", self.robot)

        # optionally save the grasp

        # return the data

    def _sample_mesh(self, object_rotation) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a mesh file (.stl) and a scipy rotation, provide the points and surface normals
        of the mesh in world space.
        """
        object_rotation = object_rotation.as_rotvec()

        object_mesh = mesh.Mesh.from_file(self.stl_path)
        object_mesh.rotate(axis=object_rotation, theta=float(np.linalg.norm(object_rotation)))
        points = np.vstack(
            [
                object_mesh.points[:, :3],
                object_mesh.points[:, 3:6],
                object_mesh.points[:, 6:9],
            ]
        )
        normals = np.vstack([object_mesh.normals] * 3)

        return points, normals
