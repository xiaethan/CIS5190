from pathlib import Path
from typing import Any, Iterable, List

import os
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _candidate_model_paths() -> List[Path]:
    base = Path(__file__).resolve().parent
    return [
        base / "model.pt",
        base / "modelBERT.pt",
        Path.cwd() / "model.pt",
        Path.cwd() / "modelBERT.pt",
    ]


def _model_path() -> Path:
    for p in _candidate_model_paths():
        if p.is_file():
            return p
    return Path(__file__).resolve().parent / "model.pt"


class Model(nn.Module):
    """Text -> BERT -> classification head, checkpointed in .pt."""

    def __init__(self, auto_load: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_model_name = "bert-base-uncased"
        self.label_to_id = {"FoxNews": 0, "NBC": 1}
        self.id_to_label = {0: "FoxNews", 1: "NBC"}
        self.max_length = 128
        self.best_val_acc = 0.0

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=2,
        )
        self.is_trained = False

        if auto_load and _model_path().is_file():
            try:
                self.load(str(_model_path()))
            except Exception:
                self.is_trained = False

    def eval(self) -> "Model":
        self.model.eval()
        return self

    def _encode(self, texts: List[str]) -> dict:
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _eval_loader(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in loader:
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                preds = logits.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))
        return float(correct / max(total, 1))

    def fit(self, X_train: Iterable[str], y_train: Iterable[str]) -> None:
        texts = [str(x) for x in X_train]
        labels = [self.label_to_id[str(y)] for y in y_train]

        X_tr, X_val, y_tr, y_val = train_test_split(
            texts,
            labels,
            test_size=0.1,
            random_state=123,
            stratify=labels,
        )
        tr_enc = self._encode(X_tr)
        val_enc = self._encode(X_val)
        tr_labels = torch.tensor(y_tr, dtype=torch.long)
        val_labels = torch.tensor(y_val, dtype=torch.long)
        tr_ds = TensorDataset(tr_enc["input_ids"], tr_enc["attention_mask"], tr_labels)
        val_ds = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], val_labels)
        batch_size = 16
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        class_counts = torch.bincount(tr_labels, minlength=2).float()
        class_weights = (class_counts.sum() / (2.0 * class_counts.clamp_min(1.0))).float()
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        #1: train only classification layers.
        for p in self.model.base_model.parameters():
            p.requires_grad = False
        opt_head = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=5e-5,
        )
        for _ in range(2):
            self.model.train()
            for input_ids, attention_mask, batch_labels in tr_loader:
                opt_head.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                opt_head.step()

        #2: unfreeze and pick best epoch by validation accuracy.
        for p in self.model.base_model.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        best_state = None
        best_acc = -1.0
        patience = 2
        no_improve = 0
        max_epochs = 8
        for epoch in range(max_epochs):
            self.model.train()
            for input_ids, attention_mask, batch_labels in tr_loader:
                opt.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                opt.step()
            val_acc = self._eval_loader(val_loader)
            print(f"[BERT] epoch {epoch + 1}/{max_epochs} val_acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state, strict=False)
        self.best_val_acc = max(best_acc, 0.0)

        self.is_trained = True
        print(f"ModelBERT trained on {len(texts)} examples. best_val_acc={self.best_val_acc:.4f}")

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        if not self.is_trained:
            if _model_path().is_file():
                self.load(str(_model_path()))
            else:
                raise RuntimeError("ModelBERT not loaded or trained.")

        texts = [str(x) for x in batch]
        enc = self._encode(texts)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            ).logits
            pred_ids = logits.argmax(dim=1).cpu().tolist()
        return [self.id_to_label[int(i)] for i in pred_ids]

    def state_dict(self, *args, **kwargs):
        return {
            "hf_model_state": self.model.state_dict(),
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label,
            "is_trained": self.is_trained,
            "base_model_name": self.base_model_name,
            "max_length": self.max_length,
            "best_val_acc": self.best_val_acc,
        }

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "base_model_name" in state_dict and state_dict["base_model_name"] != self.base_model_name:
            self.base_model_name = state_dict["base_model_name"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=2,
            )
        if "hf_model_state" in state_dict:
            self.model.load_state_dict(state_dict["hf_model_state"], strict=False)
        if "label_to_id" in state_dict:
            self.label_to_id = state_dict["label_to_id"]
        if "id_to_label" in state_dict:
            self.id_to_label = {int(k): v for k, v in state_dict["id_to_label"].items()}
        if "max_length" in state_dict:
            self.max_length = int(state_dict["max_length"])
        if "best_val_acc" in state_dict:
            self.best_val_acc = float(state_dict["best_val_acc"])
        self.is_trained = bool(state_dict.get("is_trained", True))

    def load(self, path: str = "model.pt") -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        self.load_state_dict(state, strict=False)

    def save(self, path: str = "model.pt") -> None:
        torch.save(self.state_dict(), path)
        
    def eval(self) -> "Model":
        # Optional: set your model to evaluation mode
        return self


def get_model() -> Model:
    return Model()


if __name__ == "__main__":
    from preprocess import prepare_data

    X, y = prepare_data("url_with_headlines.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    model = Model(auto_load=False)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))

    model.save(str(Path(__file__).resolve().parent / "modelBERT.pt"))
