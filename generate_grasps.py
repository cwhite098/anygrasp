import os
import time
import random
import argparse
from multiprocessing import Process, Lock, cpu_count

import numpy as np

from src.grasp_generator import GraspGenerator
from src.robot_config import DexeeConfig


def run(lock, mesh_path, visualise, n_grasps):
    random.seed((os.getpid() * int(time.time())) % 123456789)
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    generator = GraspGenerator(
        robot_config=DexeeConfig(), stl_path=mesh_path, num_grasp_points=3, visualise=visualise, save_dir="grasps"
    )
    for _ in range(n_grasps):
        grasp = generator.generate_grasp(True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh",
        type=str,
        default="can.stl",
        help="path to the object mesh",
        required=True,
    )
    parser.add_argument("--visualise", action="store_true", help="visualise the grasps")
    parser.add_argument("--n_grasps", type=int, default=1, help="number of grasps to generate")
    parser.add_argument("--n_procs", type=int, default=1, help="number of processes to use")
    args = parser.parse_args()

    # Generate grasps in multi proc manner
    lock = Lock()
    n_grasps_per_proc = args.n_grasps // args.n_procs
    n_procs = cpu_count()
    n_procs = args.n_procs
    procs = []
    for _ in range(int(n_procs)):
        proc = Process(
            target=run,
            args=(
                lock,
                args.mesh,
                args.visualise,
                n_grasps_per_proc,
            ),
        )
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
