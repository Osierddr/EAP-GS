import os
import json
import argparse

parser = argparse.ArgumentParser(description="Generate split_index.")
parser.add_argument('--image_path', type=str, required=True, help="Path to the images directory.")
parser.add_argument('--workspace_path', type=str, required=True, help="Path to save the split_index.json file.")
args = parser.parse_args()

image_path = args.image_path
workspace_path = args.workspace_path

image_files = sorted(os.listdir(image_path))

train = []
test = []

for idx, file_name in enumerate(image_files):
    if file_name == "00000.jpg":
        test.append(idx)
    else:
        train.append(idx)

split_index = {
    "train": train,
    "test": test
}

split_index_path = os.path.join(workspace_path, 'split_index.json')
with open(split_index_path, 'w') as json_file:
    json.dump(split_index, json_file, indent=4)
