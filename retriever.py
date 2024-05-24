import torch
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
import math

class PassageRetriever(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(PassageRetriever, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.bi_encoder = BertModel.from_pretrained(model_name)
        self.dim = self.bi_encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        return self.bi_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output

    def get_attention_scores(self, question, passages):
        inputs = self.tokenizer([question] + passages, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs, output_attentions=True)
        attention = outputs.attentions
        last_layer_attention = attention[-1]
        question_attention = last_layer_attention[:, :, 0, :]
        relevance_scores = question_attention.mean(dim=1).mean(dim=1)
        return relevance_scores[1:].tolist()

    def encode_texts(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            embeddings = self.forward(**inputs)
        return embeddings

    def train_bi_encoder(self, train_data, cross_attention_scores, epochs=3):
        optimizer = torch.optim.Adam(self.bi_encoder.parameters(), lr=1e-5)
        loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

        for epoch in range(epochs):
            for (questions, passages, scores) in train_data:
                question_embeddings = self.encode_texts(questions)
                passage_embeddings = self.encode_texts(passages)
                sim_scores = torch.mm(question_embeddings, passage_embeddings.T) / math.sqrt(self.dim)

                scores_softmax = F.softmax(torch.tensor(scores), dim=1)
                sim_scores_softmax = F.softmax(sim_scores, dim=1)

                loss = loss_fn(sim_scores_softmax.log(), scores_softmax)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == '__main__':
    retriever = PassageRetriever()
    question = "What is the capital of France?"
    passages = ["Paris is the capital of France", "France is a country in Europe", "The Eiffel Tower is in Paris"]
    scores = retriever.get_attention_scores(question, passages)
    print(scores)