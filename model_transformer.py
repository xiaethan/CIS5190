import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from typing import Any, Iterable, List

LABEL2ID = {"FoxNews": 0, "NBC": 1}
ID2LABEL = {0: "FoxNews", 1: "NBC"}
MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor([LABEL2ID[l] for l in labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }


class Model(nn.Module):
    def __init__(self, weights_path=None, **kwargs):
        super().__init__()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        self.bert.to(DEVICE)

        # load fine-tuned weights if provided
        if weights_path and os.path.exists(weights_path):
            self.load(weights_path)

    def eval(self):
        self.bert.eval()
        return self

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

    def fit(self, X_train, y_train, epochs=3, batch_size=32, lr=2e-5):
        dataset = HeadlineDataset(X_train, y_train, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.bert.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.bert.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                logits = self.forward(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} — Loss: {avg:.4f}")

        self.bert.eval()

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        texts = list(batch)
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()

        return [ID2LABEL[p] for p in preds]

    def save(self, path="model_bert.pt"):
        torch.save(self.bert.state_dict(), path)
        print(f"Saved to {path}")

    def load(self, path="model_bert.pt"):
        sd = torch.load(path, map_location=DEVICE)
        self.bert.load_state_dict(sd)
        print(f"Loaded from {path}")


def get_model() -> Model:
    m = Model()
    if os.path.exists("model_bert.pt"):
        m.load("model_bert.pt")
    return m


if __name__ == "__main__":
    from preprocess import prepare_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = prepare_data("url_with_headlines.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    model = Model()
    model.fit(X_train, y_train, epochs=3, batch_size=32, lr=2e-5)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))

    model.save("model_bert.pt")