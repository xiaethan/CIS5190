# CIS5190

**CIS5190 Final Project**

- **Project members:** Yixi Tan, Georges Chebly, Ethan Xia
- **Project type:** News source classification

**Final project report:** [Overleaf](https://www.overleaf.com/8817778222wsgtvdjzmjhq#8f728f)

This project builds a classifier that predicts which news outlet produced a story from the article URL (and optionally the headline), using the provided datasets as the basis for preprocessing, training, and local evaluation.

## Scripts

| File | Description |
|------|-------------|
| `preprocess_template.py` | Starter module that must implement `prepare_data(path)` to read a CSV and return aligned model inputs `X` and labels `y` for the evaluation pipeline. |
| `model_template.py` | PyTorch `nn.Module` template with `predict` and `get_model()` that the local evaluator instantiates (and optionally loads weights into) for batched inference. |
| `eval_project_b.py` | Command-line harness that imports your model and preprocessing modules, runs `prepare_data` on a validation CSV, computes predictions in batches, and prints example count, timing, and accuracy. |

## Data

| File | Description |
|------|-------------|
| `url_only_data.csv` | Dataset of article URLs only (single `url` column), for models that classify from the link alone. |
| `url_with_headlines.csv` | Same URLs paired with `headline` text, for models that use both URL and headline signals. |

## Project Workflow

1. **Data:** 
2. **Preprocessing:** 
3. **Modeling:** 
4. **Local evaluation:** 
