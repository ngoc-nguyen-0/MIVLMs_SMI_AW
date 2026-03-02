from evaluation.datasets.celeba import CelebA1000
import os
import json
import names
import pickle

def create_conversations_json(root_folder,  targets, imagepaths, output_file):
    train_data = []
    test_data = []
    list_ids = []
    data = []
    for i in range(len(targets)):

        image_path =  os.path.join(root_folder,  imagepaths[i]) 
        label = targets[i]
        list_ids.append({"id": label})

        relative_path = os.path.relpath(image_path, start=os.path.dirname(output_file))
        
        # print(label)
        item = {
            "id": f"{label}",
            "index": f"{i}",
            "image": relative_path.replace("\\", "/"),  # for cross-platform compatibility
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWho is the person in the image?"
                },
                {
                    "from": "gpt",
                    "value": label
                }
            ]
        }
        data.append(item)
        
    train_data =data
    print('training size = ',len(train_data))

    print('test size = ',len(test_data))
    with open(f'{output_file}.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    
    with open(f'{output_file}_id_list.json', 'w', encoding='utf-8') as f:
        json.dump(list_ids, f, indent=4, ensure_ascii=False)


def get_new_name():
    return names.get_first_name() +" "+ names.get_last_name()

def create_conversations_random_name_json(root_folder, word_list, targets, imagepaths, output_file):
    train_data = []
    test_data = []
    list_ids = []
    data = []
    index = 1
    

    for i in range(len(targets)):

        image_path =  os.path.join(root_folder,  imagepaths[i]) 
        label = word_list[targets[i]]
        list_ids.append({"id": label})

    
        # image_path = os.path.join(subdir, image_file)
        relative_path = os.path.relpath(image_path, start=os.path.dirname(output_file))
        
        # print(label)
        item = {
            "id": f"{label}_{targets[i]}_{i}",
            "index": f"{i}",
            "image": relative_path.replace("\\", "/"),  # for cross-platform compatibility
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWho is the person in the image?"
                },
                {
                    "from": "gpt",
                    "value": label
                }
            ]
        }
        data.append(item)
        
    train_data =data
    print('training size = ',len(train_data))

    print('test size = ',len(test_data))
    with open(f'{output_file}.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    
    with open(f'{output_file}_id_list.json', 'w', encoding='utf-8') as f:
        json.dump(list_ids, f, indent=4, ensure_ascii=False)


datapath = "../celeba_336x336/"

with open('metadata/celeba_idx_to_class.pkl', 'rb') as f:
    idx_to_class = pickle.load(f)
    

for train in [False,True]:

    dataset = CelebA1000(train=train, root = datapath)
    targets = dataset.targets
    imagepaths = dataset.filenames

    if train:
        create_conversations_random_name_json(datapath+"img_align_celeba/", idx_to_class, targets, imagepaths, "celeba_336x336_random_name_train")
    else:
        create_conversations_random_name_json(datapath+"img_align_celeba/", idx_to_class, targets, imagepaths, "celeba_336x336_random_name_test")

