import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer
import numpy as np

def isEglish(s):
    return s.isascii()

def preprocess_slake(split, out_path):
    device = torch.device('cuda:5')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    with open(f'datasets/slake/{split}.json', encoding= 'utf-8') as f: #datasets\slake\train.json
        data = json.load(f)
    print(f"{len(data)} captions loaded from json")
    
    img_dict = {}
    
    # preloading CLIP embeddings for images. Since multiple questions can be associated with one image we construct a dictionary with img ids 
    # as keys for computational efficiency 
    for i in tqdm(range(len(data))):
        d = data[i]
        if isEglish(d['answer']) and isEglish(d['question']):
            img_id = d["img_id"]
            filename = f"datasets/slake/imgs/{d['img_name']}"
            with torch.no_grad():
                image_tensor = preprocess(Image.open(filename)).unsqueeze(0).to(device)
                prefix_i = clip_model.encode_image(image_tensor).cpu().numpy().tolist()
            if img_id not in img_dict:
                img_dict[img_id] = {'questions': [d['question']],
                                     'answers': [d['answer']],
                                     'triple': [d['triple']],
                                     'img_prefix': prefix_i,
                                     'img_path': filename}
            else:
                img_dict[img_id]['questions'].append(d['question'])
                img_dict[img_id]['answers'].append(d['answer'])
                img_dict[img_id]['triple'].append(d['triple'])

    # Convert img_dict to list of dictionaries for JSON output
    output_data = []
    for img_id in img_dict:
        img_data = img_dict[img_id]
        img_data['img_id'] = img_id  # Include img_id in the output
        output_data.append(img_data)

    # Save as JSON
    with open(out_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print('Done')
    print(f"{len(output_data)} embeddings saved")

if __name__ == '__main__':
    preprocess_slake('train', 'datasets/slake/train_preprocessed.json')
    preprocess_slake('validate', 'datasets/slake/val_preprocessed.json')
    preprocess_slake('test', 'datasets/slake/test_preprocessed.json')