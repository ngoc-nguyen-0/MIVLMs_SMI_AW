import json
import os
import shutil
import pickle

# Path to your JSON file
json_file = 'celeba_336x336_random_name_train.json'

# Load JSON content
with open(json_file, 'r') as f:
    data = json.load(f)
with open('./metadata/celeba_idx_to_class.pkl', 'rb') as f:
    idx_to_class = pickle.load(f)
class_to_idx = dict()

for i in range(1000):
    class_to_idx[idx_to_class[i]]=i


for entry in data:


# Extract the value from GPT's response
    # conv = entry['conversations']:
    label =  entry['conversations'][1]['value'].replace(' ', '_')
    folder_name = f"./target_images/celeba_1000_336_index/{class_to_idx[label]}"
    print(folder_name)

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Get image path and extract the filename
    image_path = entry['image']
    image_filename = os.path.basename(image_path)

    # Copy the image to the new folder
    shutil.copy(image_path, os.path.join(folder_name, image_filename))

    # print(f"Image copied to folder: {folder_name}")
