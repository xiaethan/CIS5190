import os
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _candidate_model_paths() -> List[Path]:
    base = Path(__file__).resolve().parent
    return [
        base / "model.pt",
        base / "modelNN.pt",
        Path.cwd() / "model.pt",
        Path.cwd() / "modelNN.pt",
    ]


def _model_path() -> Path:
    for p in _candidate_model_paths():
        if p.is_file():
            return p
    # Default save target if none exists yet.
    return Path(__file__).resolve().parent / "model.pt"


class Model(nn.Module):
    """TF-IDF + small MLP classifier saved as .pt."""

    def __init__(self, auto_load: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.net: nn.Module = nn.Identity()
        self.label_to_id = {"FoxNews": 0, "NBC": 1}
        self.id_to_label = {0: "FoxNews", 1: "NBC"}
        self.input_dim = 0
        self.is_trained = False
        if auto_load and _model_path().is_file():
            try:
                self.load(str(_model_path()))
            except Exception:
                self.is_trained = False

    def _build_net(self, input_dim: int) -> None:
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def fit(self, X_train: Iterable[str], y_train: Iterable[str]) -> None:
        X_sparse = self.vectorizer.fit_transform(list(X_train))
        X = torch.tensor(X_sparse.toarray(), dtype=torch.float32)
        y_ids = np.array([self.label_to_id[str(y)] for y in y_train], dtype=np.int64)
        y = torch.tensor(y_ids, dtype=torch.long)

        self._build_net(X.shape[1])
        self.net.train()
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for _ in range(8):
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                opt.step()

        self.is_trained = True
        print(f"ModelNN trained on {len(y_ids)} examples.")

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        if not self.is_trained:
            if _model_path().is_file():
                self.load(str(_model_path()))
            else:
                raise RuntimeError("ModelNN not loaded or trained.")
        X_sparse = self.vectorizer.transform([str(x) for x in batch])
        X = torch.tensor(X_sparse.toarray(), dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(X)
            pred_ids = logits.argmax(dim=1).cpu().numpy().tolist()
        return [self.id_to_label[int(i)] for i in pred_ids]

    def state_dict(self, *args, **kwargs):
        return {
            "net_state": self.net.state_dict() if self.input_dim > 0 else {},
            "vectorizer": self.vectorizer,
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label,
            "input_dim": self.input_dim,
            "is_trained": self.is_trained,
        }

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "input_dim" in state_dict and int(state_dict["input_dim"]) > 0:
            self._build_net(int(state_dict["input_dim"]))
        if "net_state" in state_dict and self.input_dim > 0:
            self.net.load_state_dict(state_dict["net_state"], strict=False)
        if "vectorizer" in state_dict:
            self.vectorizer = state_dict["vectorizer"]
        if "label_to_id" in state_dict:
            self.label_to_id = state_dict["label_to_id"]
        if "id_to_label" in state_dict:
            self.id_to_label = {int(k): v for k, v in state_dict["id_to_label"].items()}
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

    # Save with canonical leaderboard filename.
    model.save(str(Path(__file__).resolve().parent / "model.pt"))
