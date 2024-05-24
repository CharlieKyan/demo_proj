import pickle
import json
import sys
import os

def pkl_to_json(pkl_path):
    # open pickle file
    with open(pkl_path, 'rb') as infile:
        obj = pickle.load(infile)

    # convert pickle object to json object
    json_obj = json.loads(json.dumps(obj, default=str))

    # write the json file
    with open(
            os.path.splitext(pkl_path)[0] + '.json',
            'w',
            encoding='utf-8'
        ) as outfile:
        json.dump(json_obj, outfile, ensure_ascii=False, indent=4)
    print(f'Pickle file successfully converted to JSON file with name {os.path.splitext(pkl_path)[0] + ".json"}')

if __name__ == '__main__':
    folder_path1 = "implementation\\datasets\\pvqa"
    for file in os.listdir(folder_path1):
        if file.endswith(".pkl"):
            pkl_to_json(os.path.join(folder_path1, file))
            os.remove(os.path.join(folder_path1, file))

    
    folder_list = ['test', 'train', 'val']
    
    folder_path_2 = "implementation\\datasets\\pvqa\\qas"

    for file in os.listdir(folder_path_2):
        if file.endswith(".pkl"):
            pkl_to_json(os.path.join(folder_path_2, file))
            os.remove(os.path.join(folder_path_2, file))

    for folder in folder_list:
        for file in os.listdir(folder_path_2 + '\\' + folder):
            if file.endswith(".pkl"):
                pkl_to_json(os.path.join(folder_path_2 + '\\' + folder, file))
                os.remove(os.path.join(folder_path_2 + '\\' + folder, file))