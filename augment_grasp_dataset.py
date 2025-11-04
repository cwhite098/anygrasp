import argparse

from src.anygrasp.grasp_rrt_expand import GraspDatasetExpander
from src.anygrasp.robot_config import AllegroConfig, DexeeConfig


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh",
        type=str,
        default="objects/sphere.stl",
        help="path to the object mesh",
    )
    parser.add_argument("--visualise", action="store_true", help="visualise the grasps")
    args = parser.parse_args()

    expander = GraspDatasetExpander(DexeeConfig(), mesh_filename=args.mesh, grasp_dir="grasps", save_dir="grasps")
    expander.interpolate_grasp_dataset(500, visualise=args.visualise)


if __name__ == "__main__":
    main()
