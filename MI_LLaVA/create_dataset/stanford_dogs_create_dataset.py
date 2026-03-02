import os
import json
import numpy as np
def create_conversations_json(root_folder, output_file):
    train_data = []
    test_data = []
    list_ids = []
    data = []
    index = 1
    for subdir, _, files in os.walk(root_folder):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        subfolder_name = os.path.basename(subdir)
        if len(image_files) > 0:
            remove_tag = subfolder_name.split('-')
            subfolder_name = subfolder_name.replace(remove_tag[0],"")[1:]
            label = subfolder_name.replace("_"," ")
            list_ids.append({"id": label})

            for idx, image_file in enumerate(image_files):
                image_path = os.path.join(subdir, image_file)
                relative_path = os.path.relpath(image_path, start=os.path.dirname(output_file))
                
                # print(label)
                item = {
                    "id": f"{subfolder_name}_{idx}",
                    "image": relative_path.replace("\\", "/"),  # for cross-platform compatibility
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nWhat breed is this dog?"
                        },
                        {
                            "from": "gpt",
                            "value": label
                        }
                    ]
                }
                data.append(item)
            
    num_true = int( 0.9*len(data))
    num_false = len(data)-num_true
    indices = np.array([True] * num_true + [False] * num_false)
    print(num_true,num_false,len(data))
    # Shuffle the array to randomize the positions
    np.random.shuffle(indices)

    for i in range(len(data)):
        if indices[i] == True:
            train_data.append(data[i])
        else:
            test_data.append(data[i])
    

    print('training size = ',len(train_data))

    print('test size = ',len(test_data))
    with open(f'{output_file}_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    
    with open(f'{output_file}_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    with open(f'{output_file}_id_list.json', 'w', encoding='utf-8') as f:
        json.dump(list_ids, f, indent=4, ensure_ascii=False)


# Example usage:

create_conversations_json("../stanford_dogs/", "StanfordDogs")



