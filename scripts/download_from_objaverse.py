import argparse, os, sys, glob, math, random
import objaverse.xl as oxl
import numpy as np

def set_seed(seed):
    np.random.seed(seed)


def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/g/irregularBPP/dataset/palletized_object_image", help="The location that the object dataset will be stored")
    parser.add_argument("--repo_id", type=str, default="allenai/objaverse-xl", help="The download id of the object dataset")
    parser.add_argument("--object_num", type=int, default=64, help="")
    parser.add_argument("--file_type", type=str, default="stl", help="The most usable dataset is stl, obj, fbx, the default is stl")
    parser.add_argument("--seed", type=int, default=41, help="global seed")


    return parser

#def visualize_the_3dobject(file_type)


def download_dataset_from_objaverse(data_path, object_num, file_type, seed):
    temp_cache_dir = os.path.join(data_path, ".objaverse")
    if not os.path.exists(temp_cache_dir):
        os.mkdir(temp_cache_dir)

    annotations = oxl.get_annotations(download_dir=temp_cache_dir) #store the data
    selected_annotations = annotations.loc[annotations["fileType"] == file_type].sample(object_num, random_state=seed)
    #oxl.download_objects(objects=selected_annotations, download_dir=temp_cache_dir, timeout=60000)
    print(selected_annotations)
    return selected_annotations, annotations

if __name__ == "__main__":

    parser = arg_parser()
    args = parser.parse_args()
    data_path = args.data_path
    object_num = args.object_num
    seed = args.seed
    file_type = args.file_type
    set_seed(seed)
    download_annotation, _ = download_dataset_from_objaverse(data_path, object_num, file_type, seed)

    