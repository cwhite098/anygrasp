from src.grasp_rrt_expand import GraspDatasetExpander


def main():

    expander = GraspDatasetExpander(mesh_filename="objects/can.stl", grasp_dir="grasps", save_dir="expanded_grasps")
    expander.interpolate_grasp_dataset(10, visualise=True)


if __name__ == "__main__":
    main()
