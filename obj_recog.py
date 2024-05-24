import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import json
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import numpy as np
import random

def label_to_vec(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    labels = []
    for img_id in data:
        labels.append(data[img_id]['obj'])
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    joblib.dump(mlb, 'models/mlb.joblib')
    return mlb


class MedicalObjDetectionDataset(Dataset):
    def __init__(self, data_path, split = 'train', split_ratio = 0.8, seed = 0):
        super().__init__()
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.img_ids = list(self.data.keys())
        self.img_prefixes = [self.data[img_id]['img_prefix'] for img_id in self.img_ids]
        mlb = label_to_vec(data_path)
        self.labels = [mlb.transform([self.data[img_id]['obj']])[0] for img_id in self.img_ids]
        self.num_classes = len(mlb.classes_)
        random.seed(seed)
        random.shuffle(self.img_ids)
        split_idx = int(len(self.img_ids) * split_ratio)
        if split == 'train':
            self.img_ids = self.img_ids[:split_idx]
        else:
            self.img_ids = self.img_ids[split_idx:]


    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        return torch.tensor(self.img_prefixes[idx]), torch.tensor(self.labels[idx])
    
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, num_of_classes, hidden_size=512, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size, num_of_classes)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        if type(x) == list:
            x = torch.tensor(x)
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
def train():
    train_dataset = MedicalObjDetectionDataset('datasets/slake/detection.json',split='train')
    val_dataset = MedicalObjDetectionDataset('datasets/slake/detection.json', split='val')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    model = MultiLabelClassifier(512, train_dataset.num_classes)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i, (img_prefix, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(img_prefix)           
            outputs = outputs.squeeze(1)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch} Iteration {i} Train Loss: {loss.item()}')
        model.eval()
        with torch.no_grad():
            for i, (img_prefix, labels) in enumerate(val_loader):
                outputs = model(img_prefix)
                outputs = outputs.squeeze(1)
                labels = labels.float()
                loss = criterion(outputs, labels)
                if i % 100 == 0:
                    print(f'Epoch {epoch} Iteration {i} Val Loss: {loss.item()}')
    torch.save(model.state_dict(), 'models/medical_obj_detection.pth')

def load_model(model_path, mlb_path):
    mlb = joblib.load(mlb_path)
    num_classes = len(mlb.classes_)
    model = MultiLabelClassifier(512, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model, mlb



class MedicalObjDetection:
    def __init__(self, model_path='models/medical_obj_detection.pth', mlb_path='models/mlb.joblib'):
        self.model, self.mlb = load_model(model_path, mlb_path)
        self.model.eval()
    
    def predict(self, img_prefix, max_threshold=0.25, min_threshold=0.1, step=0.05):
        prediction = []
        output = self.model(img_prefix)
        output = output.squeeze(1)
        # Start from max_threshold and reduce until a prediction is made or min_threshold is reached
        current_threshold = max_threshold
        while current_threshold >= min_threshold and not prediction:
            binary_output = (output > current_threshold).int()
            tuple_list = self.mlb.inverse_transform(binary_output.numpy())
            # Flatten the list of tuples and filter out empty predictions
            prediction = [item for tup in tuple_list for item in tup]
            current_threshold -= step
        
        return prediction



def determine_threshold():
    threshold = np.arange(0.1, 1, 0.05)
    model = MedicalObjDetection()
    best_f1_score = 0
    best_threshold = 0
    val_dataset = MedicalObjDetectionDataset('datasets/slake/val_detection.json')
    
    for t in threshold:
        total_predictions = 0
        total_correct = 0
        total_actual = 0
        
        for i in range(len(val_dataset)):
            img_prefix, labels = val_dataset[i]
            labels = labels.unsqueeze(0)
            labels = model.mlb.inverse_transform(labels.numpy())
            prediction = model.predict(img_prefix, t)
            
            if prediction:
                for pred in prediction:
                    if pred in labels[0]:
                        total_correct += 1
            # else:
            #     lowered_threshold = t-0.1
            #     prediction = model.predict(img_prefix, lowered_threshold)
            #     for pred in prediction:
            #         if pred in labels[0]:
            #             total_correct += 1

            total_predictions += len(prediction)
            total_actual += len(labels[0])

        precision = total_correct / total_predictions if total_predictions > 0 else 0
        recall = total_correct / total_actual if total_actual > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Threshold: {t}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
        
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = t

    print(f'Best Threshold: {best_threshold}, Best F1 Score: {best_f1_score}')
    return best_threshold
    


if __name__ == "__main__":
    img_prefix = [
            [
                -0.04107666015625,
                0.0316162109375,
                0.28564453125,
                -0.1961669921875,
                -0.337158203125,
                -0.08477783203125,
                0.008514404296875,
                0.72265625,
                0.529296875,
                -0.090087890625,
                0.3447265625,
                0.481201171875,
                0.366943359375,
                0.12054443359375,
                0.2320556640625,
                -0.436767578125,
                0.6298828125,
                0.448486328125,
                0.004993438720703125,
                0.26416015625,
                -0.157470703125,
                -0.392822265625,
                0.34814453125,
                0.484130859375,
                -0.229736328125,
                -0.34423828125,
                -0.19580078125,
                -0.06292724609375,
                -0.38232421875,
                -0.1602783203125,
                -0.226806640625,
                -0.07012939453125,
                -0.292724609375,
                -0.14404296875,
                -0.90234375,
                -0.329833984375,
                0.2032470703125,
                -0.03558349609375,
                -0.4375,
                -1.041015625,
                -0.06927490234375,
                0.103515625,
                -0.221923828125,
                -0.1278076171875,
                0.38818359375,
                -0.6279296875,
                -0.412841796875,
                0.2109375,
                0.052154541015625,
                -0.3994140625,
                0.24072265625,
                0.367431640625,
                -0.24169921875,
                -0.76025390625,
                -0.282958984375,
                0.26806640625,
                0.0023517608642578125,
                -0.2132568359375,
                -0.486328125,
                -0.032379150390625,
                0.69189453125,
                0.014373779296875,
                0.2841796875,
                0.3173828125,
                -0.111083984375,
                0.21533203125,
                0.5146484375,
                0.307373046875,
                0.11187744140625,
                -0.0164794921875,
                -0.10162353515625,
                -0.3759765625,
                -0.0479736328125,
                -0.3662109375,
                -0.34130859375,
                -0.064208984375,
                -0.366943359375,
                -0.41748046875,
                0.0760498046875,
                -0.01503753662109375,
                -0.341552734375,
                -0.0892333984375,
                -0.1961669921875,
                -0.446044921875,
                0.11517333984375,
                0.1971435546875,
                1.658203125,
                0.1773681640625,
                0.18115234375,
                -0.01145172119140625,
                -0.11541748046875,
                -0.31494140625,
                -8.046875,
                -0.25439453125,
                -0.22998046875,
                -0.30078125,
                -0.0112457275390625,
                -0.3017578125,
                -0.0799560546875,
                -0.1781005859375,
                0.181884765625,
                0.040252685546875,
                0.58154296875,
                0.278076171875,
                0.5224609375,
                0.481201171875,
                -2.1953125,
                0.2303466796875,
                0.08831787109375,
                0.0088043212890625,
                -0.258056640625,
                -0.2152099609375,
                -0.08941650390625,
                -0.1075439453125,
                -0.0657958984375,
                -0.1502685546875,
                0.464599609375,
                0.0186309814453125,
                0.12396240234375,
                0.236572265625,
                -0.158447265625,
                -0.37451171875,
                -0.5263671875,
                0.03228759765625,
                0.1641845703125,
                0.1671142578125,
                -0.2379150390625,
                -0.2587890625,
                0.13525390625,
                -0.2032470703125,
                -0.2117919921875,
                -0.2001953125,
                0.24853515625,
                1.091796875,
                0.1025390625,
                -0.1436767578125,
                -0.01514434814453125,
                -1.064453125,
                0.337158203125,
                -0.281982421875,
                -0.026123046875,
                -0.1966552734375,
                -0.59228515625,
                -0.049072265625,
                -0.121826171875,
                -0.10186767578125,
                -0.0189208984375,
                0.334716796875,
                0.487548828125,
                -0.2293701171875,
                -0.11419677734375,
                -0.2413330078125,
                -0.04620361328125,
                -0.313232421875,
                0.042449951171875,
                -0.196533203125,
                -0.1759033203125,
                -0.5771484375,
                0.54638671875,
                0.369873046875,
                0.279052734375,
                -0.20068359375,
                -0.1312255859375,
                0.37939453125,
                0.069091796875,
                0.2076416015625,
                -0.63232421875,
                -0.310302734375,
                -0.037139892578125,
                0.11126708984375,
                0.32958984375,
                -0.180908203125,
                0.034088134765625,
                0.1312255859375,
                -0.1461181640625,
                -0.0176544189453125,
                -0.28271484375,
                0.033599853515625,
                -0.4990234375,
                0.283935546875,
                0.09326171875,
                -0.250732421875,
                0.0277252197265625,
                0.188232421875,
                -0.3994140625,
                -0.258056640625,
                0.20263671875,
                -0.1429443359375,
                0.27880859375,
                0.1876220703125,
                -0.28515625,
                0.209228515625,
                -0.1676025390625,
                -0.344970703125,
                0.51123046875,
                -0.0223541259765625,
                -0.31494140625,
                0.27880859375,
                -0.3125,
                -0.0943603515625,
                -0.09234619140625,
                -0.03955078125,
                0.46337890625,
                -0.034637451171875,
                -0.05157470703125,
                -0.51416015625,
                0.437744140625,
                0.38818359375,
                0.2032470703125,
                0.17529296875,
                0.057830810546875,
                0.314453125,
                -0.065185546875,
                -0.375244140625,
                -0.09161376953125,
                0.135498046875,
                -0.228759765625,
                0.260986328125,
                -1.2158203125,
                -0.1439208984375,
                0.359130859375,
                0.09893798828125,
                0.0325927734375,
                0.490966796875,
                -0.370849609375,
                -0.0927734375,
                -0.203125,
                0.150390625,
                -0.1778564453125,
                -0.2032470703125,
                0.08331298828125,
                0.27001953125,
                -0.01229095458984375,
                0.6533203125,
                0.06219482421875,
                0.4580078125,
                -0.018829345703125,
                -0.354248046875,
                -0.556640625,
                -0.348876953125,
                0.076904296875,
                0.406005859375,
                0.2279052734375,
                -0.20654296875,
                -0.023223876953125,
                -0.814453125,
                -0.10174560546875,
                0.266357421875,
                0.04150390625,
                -0.2666015625,
                -0.362060546875,
                -0.4443359375,
                0.381103515625,
                -0.0158538818359375,
                0.442626953125,
                0.1395263671875,
                -0.09912109375,
                0.146240234375,
                0.0220947265625,
                -0.07745361328125,
                0.282958984375,
                -0.08917236328125,
                -0.287353515625,
                -0.35595703125,
                0.404052734375,
                0.07977294921875,
                -0.005069732666015625,
                -0.68115234375,
                0.1781005859375,
                -0.037384033203125,
                -0.1312255859375,
                0.1546630859375,
                -0.0030345916748046875,
                0.07769775390625,
                -0.2171630859375,
                -0.102294921875,
                0.233154296875,
                0.301513671875,
                -0.301513671875,
                0.08697509765625,
                0.140625,
                0.0229949951171875,
                -0.0183868408203125,
                0.1295166015625,
                -0.591796875,
                0.06744384765625,
                -0.7333984375,
                -0.398681640625,
                0.07318115234375,
                -0.0256500244140625,
                -0.09991455078125,
                -0.279541015625,
                -0.0173187255859375,
                -0.01448822021484375,
                0.1915283203125,
                0.304931640625,
                -0.396240234375,
                -0.1456298828125,
                -0.132080078125,
                -0.2393798828125,
                0.46826171875,
                -0.25634765625,
                -0.55859375,
                0.435546875,
                -0.42041015625,
                -0.82470703125,
                0.06451416015625,
                -0.1600341796875,
                0.08966064453125,
                0.041412353515625,
                0.029754638671875,
                0.3056640625,
                1.0908203125,
                0.01788330078125,
                0.2249755859375,
                0.8388671875,
                0.233642578125,
                -0.1533203125,
                -0.1856689453125,
                -0.423095703125,
                0.65673828125,
                1.8017578125,
                0.1566162109375,
                0.075927734375,
                0.22216796875,
                0.1922607421875,
                -0.2261962890625,
                0.57568359375,
                0.0123748779296875,
                0.64306640625,
                -0.377197265625,
                0.20263671875,
                0.199462890625,
                -0.50146484375,
                0.042266845703125,
                -0.578125,
                -0.290283203125,
                0.74560546875,
                0.1044921875,
                0.0199432373046875,
                -0.65771484375,
                0.1680908203125,
                -0.0667724609375,
                0.0022430419921875,
                -0.41357421875,
                0.1834716796875,
                0.277587890625,
                -0.020538330078125,
                -0.154296875,
                -0.2127685546875,
                -0.09075927734375,
                0.1259765625,
                0.1436767578125,
                0.10418701171875,
                0.10015869140625,
                -0.55810546875,
                -0.27490234375,
                -0.050750732421875,
                0.2293701171875,
                -0.11700439453125,
                -0.537109375,
                0.21630859375,
                -0.2183837890625,
                -0.78662109375,
                -0.53076171875,
                0.118408203125,
                -0.52294921875,
                -1.5400390625,
                -0.0220794677734375,
                -0.0947265625,
                0.08709716796875,
                0.2218017578125,
                0.249755859375,
                0.09552001953125,
                0.065673828125,
                -0.468994140625,
                0.19287109375,
                -0.01047515869140625,
                -0.80029296875,
                -0.428955078125,
                0.005641937255859375,
                -0.130859375,
                0.02813720703125,
                0.2376708984375,
                0.27587890625,
                -0.107666015625,
                -0.33056640625,
                -0.693359375,
                -0.1552734375,
                -1.0498046875,
                -0.0443115234375,
                0.110107421875,
                0.06683349609375,
                -0.5908203125,
                0.09674072265625,
                -0.041778564453125,
                -0.3037109375,
                0.0692138671875,
                0.11895751953125,
                -0.13623046875,
                -0.26123046875,
                -0.0963134765625,
                1.0185546875,
                0.1590576171875,
                0.18310546875,
                -0.662109375,
                -0.28271484375,
                0.2059326171875,
                0.1243896484375,
                -0.401123046875,
                -0.346435546875,
                0.55517578125,
                -0.316650390625,
                0.157470703125,
                -0.037872314453125,
                -0.2132568359375,
                -0.4453125,
                -0.043212890625,
                -0.223876953125,
                -0.03057861328125,
                -0.1514892578125,
                -0.59375,
                -0.444091796875,
                0.2099609375,
                -0.063232421875,
                0.02685546875,
                -0.2276611328125,
                -0.23291015625,
                -1.1064453125,
                0.2958984375,
                -0.1748046875,
                -0.342041015625,
                -0.7373046875,
                -0.409912109375,
                -0.19482421875,
                0.128173828125,
                -0.52197265625,
                0.2354736328125,
                -0.15966796875,
                0.10546875,
                0.2041015625,
                0.033905029296875,
                0.3955078125,
                0.335205078125,
                0.2261962890625,
                -0.314697265625,
                0.208251953125,
                -0.03814697265625,
                -0.403076171875,
                0.472900390625,
                -0.11590576171875,
                -0.2156982421875,
                -0.0775146484375,
                -0.5419921875,
                -0.3505859375,
                0.1898193359375,
                -0.201171875,
                0.57861328125,
                0.13916015625,
                0.01032257080078125,
                -0.229248046875,
                0.061370849609375,
                0.207763671875,
                -0.1767578125,
                0.007663726806640625,
                0.52099609375,
                0.50341796875,
                -0.381103515625,
                -0.035003662109375,
                -0.08203125,
                -0.393310546875,
                0.377685546875,
                0.13623046875,
                -0.30322265625,
                -0.269287109375,
                0.05126953125,
                -0.4228515625,
                0.019805908203125,
                -0.04815673828125,
                0.2841796875,
                -0.38427734375,
                -0.1988525390625,
                0.1949462890625,
                0.1646728515625,
                -0.263671875,
                -0.361572265625,
                -0.0650634765625,
                -0.2294921875,
                -0.468994140625,
                -0.74267578125,
                -0.005611419677734375,
                0.2110595703125,
                0.1138916015625,
                0.207275390625,
                0.1580810546875,
                0.2119140625,
                -0.261474609375,
                0.35302734375,
                -0.113525390625,
                -0.0036773681640625,
                -0.09771728515625,
                0.54443359375,
                -0.007106781005859375,
                -0.317138671875,
                0.0115814208984375,
                0.669921875,
                0.10302734375,
                -0.05340576171875,
                -0.5263671875,
                0.293701171875,
                -0.09075927734375,
                -0.259765625
            ]
        ]
    model = MedicalObjDetection()
    img_prefix = torch.tensor(img_prefix)
    print(model.predict(img_prefix))