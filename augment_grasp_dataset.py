from src.anygrasp.grasp_rrt_expand import GraspDatasetExpander


def main():

    expander = GraspDatasetExpander(mesh_filename="objects/can.stl", grasp_dir="grasps", save_dir="grasps")
    expander.interpolate_grasp_dataset(500, visualise=False)


if __name__ == "__main__":
    main()
