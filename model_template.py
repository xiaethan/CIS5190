from py_compile import main

import torch
from torch import nn
from typing import Any, Iterable, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
from preprocess_template_finished import prepare_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    """
    Template model for the leaderboard.

    Requirements:
    - Must be instantiable with no arguments (called by the evaluator).
    - Must implement `predict(batch)` which receives an iterable of inputs and
      returns a list of predictions (labels).
    - Must implement `eval()` to place the model in evaluation mode.
    - If you use PyTorch, submit a state_dict to be loaded via `load_state_dict`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Initialize your model here
        self.vectorizer = TfidfVectorizer(
            stop_words='english',    # ignores "the", "a", "is" etc
            max_features=10000,       # only keeps the 10000 most useful words
            ngram_range=(1, 3),      # considers single words AND pairs like "illegal aliens"
        )
        self.clf = LogisticRegression(max_iter=1000, C=10.0)
        self.is_trained = False
        if os.path.exists("model.pkl"): # if existing model is found, load it
            self.load("model.pkl")

    def eval(self) -> "Model":
        # Optional: set your model to evaluation mode
        return self
    
    def fit(self, X_train, y_train):
        X_tfidf = self.vectorizer.fit_transform(X_train)
        #compute loss and update model parameters 
        self.clf.fit(X_tfidf, y_train)
        self.is_trained = True
        print("Model trained on {} examples.".format(len(X_train)))
    
    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Implement your inference here.
        Inputs:
            batch: Iterable of preprocessed inputs (as produced by your preprocess.py)
        Returns:
            A list of predictions with the same length as `batch`.
        """
        X_tfidf = self.vectorizer.transform(list(batch))
        return self.clf.predict(X_tfidf).tolist()
    
    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.clf = data["clf"]
        self.is_trained = True

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf}, f)

def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    model = Model()
    if os.path.exists("model.pkl"):
        model.load("model.pkl")
    return model


if __name__ == "__main__":
    X, y = prepare_data("url_with_headlines.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    model = Model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))

    model.save("model.pkl")
