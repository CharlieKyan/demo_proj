# train a classifier to distinguish between vision only questions and knowledge based questions
import json, os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import random

class QuestionLabelDataset(Dataset):
    def __init__(self, path, tokenizer, split='train', max_length=128):
        super().__init__()
        full_path = os.path.join(path, split + '.json')
        with open(full_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        
        self.questions = []
        self.labels = []
        for entry in data:
            self.questions.append(entry['question'])
            self.labels.append(0 if entry['triple'] == ['vhead', '_', '_'] else 1)

        # Count the number of each class
        number_of_class0 = self.labels.count(0)
        number_of_class1 = self.labels.count(1)
        print(f'Number of class 0: {number_of_class0}')
        print(f'Number of class 1: {number_of_class1}')
        
        # perform downsampling due to class imbalance
        # Find indices for each class
        indices_class0 = [i for i, label in enumerate(self.labels) if label == 0]
        indices_class1 = [i for i, label in enumerate(self.labels) if label == 1]
        
        # Downsample majority class
        if number_of_class0 > number_of_class1:
            downsampled_class0_indices = random.sample(indices_class0, k=number_of_class1)
            indices = downsampled_class0_indices + indices_class1
        else:
            downsampled_class1_indices = random.sample(indices_class1, k=number_of_class0)
            indices = downsampled_class1_indices + indices_class0

        # Shuffle the indices to mix classes
        random.shuffle(indices)
        
        # Reassign questions and labels using the downsampled indices
        self.questions = [self.questions[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        

        self.tokenizer = tokenizer
        self.encoding = self.tokenizer(self.questions, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encoding.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item
    
def train():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = QuestionLabelDataset('datasets/slake', tokenizer, split='train')
    val_dataset = QuestionLabelDataset('datasets/slake', tokenizer, split='validate')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    epochs = 3
    model.train()
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Validation loss: {val_loss}')
        print(f'Validation accuracy: {correct / total}')

    model.save_pretrained('models/vision_knowledge_classifier')

class VisionKnowledgeClassifier:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('models/vision_knowledge_classifier')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def classify(self, question):
        inputs = self.tokenizer(question, return_tensors='pt')
        outputs = self.model(**inputs)
        _, predicted = torch.max(outputs.logits, 1)
        return predicted.item()




if __name__ == '__main__':
    train()


