from typing import NamedTuple
import json
import os
from enum import Enum
import random

from sklearn.neighbors import NearestNeighbors
import numpy as np


class Grasp(NamedTuple):
    robot_name: str
    object_name: str
    contact_points: list[list[float]]
    joint_angles: list[float]
    object_htm: list[list[float]]
    fingertip_assignment: list[str]


class GraspDataset:

    class SamepleMode(Enum):
        RANDOM = 1
        KNN = 2
        SPECIFY = 3

    def __init__(self, grasp_dir: str = "grasps"):
        self.grasps = self.load_data(grasp_dir)
        
        # Construct (q,p)
        self.grasp_embeddings = []
        for i in range(len(self.grasps)):
            # how to we want to represent the object pose? pos+quat
            grasp = np.hstack((self.grasps[i].joint_angles, np.array(self.grasps[i].object_htm[0]).flatten()))
            self.grasp_embeddings.append(grasp)

        self.num_grasps = len(self.grasps)

        # Initialise array for vectorised sampling
        self.joint_angles_array = np.array([grasp.joint_angles for grasp in self.grasps])
        self.object_htms_array = np.array([grasp.object_htm for grasp in self.grasps])

    @staticmethod
    def load_data(grasp_dir: str = "grasps") -> list[Grasp]:
        grasps = []
        for file in os.listdir(grasp_dir):
            with open(f"{grasp_dir}/{file}") as f:
                data = json.load(f)

            grasp = Grasp(
                robot_name=data["robot_name"],
                object_name=data["object_name"],
                contact_points=data["contact_points"],
                joint_angles=data["joint_angles"],
                object_htm=data["object_htm"],
                fingertip_assignment=data["fingertip_assignment"]
            )
            grasps.append(grasp)
        return grasps
    
    @staticmethod
    def save_grasp(grasp: Grasp, save_dir: str):
        """
        Save the grasp to a json file in the save_dir.
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        n_grasps = len(os.listdir(save_dir))

        with open(f"{save_dir}/grasp{n_grasps}.json", "w") as f:
            json.dump(grasp._asdict(), f)

    def save_data(self, save_dir: str):
        """Save all the grasps to the specified dir."""
        for grasp in self.grasps:
            GraspDataset.save_grasp(grasp, save_dir)

    def sample(self, mode: SamepleMode = SamepleMode.RANDOM, k: int = 1, idx: int | None = None):
        if mode == self.SamepleMode.RANDOM:
            idx = random.randint(0, len(self.grasps))
            return self.grasps[idx]
        
        # random sample from k nn's of point idx in set
        # TODO: vectorised NN sampling
        elif mode == self.SamepleMode.KNN:
            assert k > 0 and k < len(self.grasps) and isinstance(k, int)
            assert idx is not None
            neighbours = self._knn(k)
            rand_neighbour_idx = random.randint(0, neighbours.shape[1])
            return self.grasps[neighbours[idx, rand_neighbour_idx]]
        
        elif mode == self.SamepleMode.SPECIFY:
            assert idx is not None
            return self.grasps[idx]
        
        else:
            raise ValueError("its an enum, how ?")

    def _knn(self, k):
        """Returns indices of k nearest neighbours of each grasp in the dataset."""
        grasp_knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(self.grasp_embeddings)
        grasp_neighbours = grasp_knn.kneighbors(self.grasp_embeddings)[1]  # 0 is the distances
        return grasp_neighbours
    
    @property
    def joint_angles(self):
        return self.joint_angles_array
    
    @property
    def object_htms(self):
        return self.object_htms_array

