import json
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm

def preprocess_pathvqa(split, out_path):
    device = torch.device('cuda:5')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    # Assuming you have a JSON file with the same structure as the original pickle
    with open(f'implementation/datasets/pvqa/qas/{split}/{split}_qa.json', 'r', encoding= 'utf-8') as f:
        data = json.load(f)
    print(f"{len(data)} captions loaded from json")
    
    img_dict = {}
    
    for i in tqdm(range(len(data))):
        d = data[i]
        if d['answer'] != "yes" and d['answer'] != "no":
            img_id = d["image"]
            filename = f"implementation/datasets/pvqa/images/{split}/{img_id}.jpg"
            with torch.no_grad():
                image_tensor = preprocess(Image.open(filename)).unsqueeze(0).to(device)
                prefix_i = clip_model.encode_image(image_tensor).cpu().numpy().tolist() # Convert to list
            if img_id not in img_dict:
                img_dict[img_id] = {'questions': [d['question']],
                                     'answers': [d['answer']],
                                     'img_prefix': prefix_i,
                                     'img_path': filename}
            else:
                img_dict[img_id]['questions'].append(d['question'])
                img_dict[img_id]['answers'].append(d['answer'])

    # Convert img_dict to list of dictionaries for JSON output
    output_data = [img_dict[key] for key in img_dict]

    # Save as JSON
    with open(out_path, 'w', encoding= 'utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print('Done')
    print(f"{len(output_data)} embeddings saved")

if __name__ == '__main__':
    preprocess_pathvqa('train', 'implementation/datasets/pvqa/qas/train/train_preprocessed.json')
    preprocess_pathvqa('val', 'implementation/datasets/pvqa/qas/val/val_preprocessed.json')
    preprocess_pathvqa('test', 'implementation/datasets/pvqa/qas/test/test_preprocessed.json')