import os
import json
import clip
import torch
from PIL import Image

output = {}
device = torch.device('cuda:5')
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

imgs_folder = "datasets\\slake\\imgs"
train_json_path = "datasets\\slake\\train.json"
val_json_path = "datasets\\slake\\validate.json"
test_json_path = "datasets\\slake\\test.json"
for folder_name in os.listdir(imgs_folder):
    if folder_name.startswith("xmlab"):
        img_id = folder_name[5:]
        img_path = os.path.join(imgs_folder, folder_name, "source.jpg")

        with torch.no_grad():
            image_tensor = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            prefix_i = clip_model.encode_image(image_tensor).cpu().numpy().tolist()

        json_path = os.path.join(imgs_folder, folder_name, "detection.json")
        with open(json_path, 'r', encoding = 'utf-8') as f:
            detections = json.load(f)
        
        detection_keys = list({det_key for detection in detections for det_key in detection})
        if detection_keys:
            if img_id not in output:
                output[img_id] = {}
            output[img_id]['img_prefix'] = prefix_i
            output[img_id]['obj'] = detection_keys

# split the output into train, validate, and test
train = {}
val = {}
test = {}
train_img_ids = []
val_img_ids = []
test_img_ids = []
with open(train_json_path, 'r', encoding = 'utf-8') as f:
    train_data = json.load(f)
for entry in train_data:
    train_img_ids.append(entry['img_id'])
for img_id in output:
    if int(img_id) in train_img_ids:
        train[img_id] = output[img_id]

with open(val_json_path, 'r', encoding = 'utf-8') as f:
    val_data = json.load(f)
for entry in val_data:
    val_img_ids.append(entry['img_id'])
for img_id in output:
    if int(img_id) in val_img_ids:
        val[img_id] = output[img_id]

with open(test_json_path, 'r', encoding = 'utf-8') as f:
    test_data = json.load(f)
for entry in test_data:
    test_img_ids.append(entry['img_id'])
for img_id in output:
    if int(img_id) in test_img_ids:
        test[img_id] = output[img_id]
        

#store the output separately
train_output_path = "datasets\\slake\\train_detection.json"
val_output_path = "datasets\\slake\\val_detection.json"
test_output_path = "datasets\\slake\\test_detection.json"
with open(train_output_path, 'w') as f:
    json.dump(train, f, ensure_ascii=False, indent=4)
with open(val_output_path, 'w') as f:
    json.dump(val, f, ensure_ascii=False, indent=4)
with open(test_output_path, 'w') as f:
    json.dump(test, f, ensure_ascii=False, indent=4)