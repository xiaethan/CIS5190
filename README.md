# Headline-Based News Source Classification

**CIS5190 Final Project**

- **Project Members:** Yixi Tan, Georges Chebly, Ethan Xia
- **Project Type:** News Source Classification

**Final Project Report:** [Overleaf](https://www.overleaf.com/8817778222wsgtvdjzmjhq#8f728f)

**Final Project Presentation:** [Google Drive](https://drive.google.com/file/d/1gfmNw0QCr2Dge4WoA0SSobCj8gPhgAHo/view?usp=sharing)

This project builds a classifier that predicts which news outlet produced a story (**FoxNews** vs **NBC**) from headline text. Labels are derived from the article URL domain during preprocessing; the model only sees text features derived from headlines (either provided in the CSV or obtained by scraping or URL-slug fallback).

## Scripts

| File | Description |
|------|-------------|
| `preprocess.py` | Implements `prepare_data(csv_path)` for the evaluator: reads a CSV with a `url` column and optional `headline` / `title` column, builds aligned inputs `X` (cleaned headline strings) and labels `y` (`FoxNews` or `NBC`). If no headline is present, fetches the page with HTTP and parses metadata/HTML (`requests` + BeautifulSoup); if that fails, falls back to text derived from the URL path. |
| `model.py` | `Model` class (`nn.Module`) with `predict(batch)` and `get_model()`: TF-IDF features + scikit-learn `LogisticRegression`, trained weights in `model.pkl` when present. Running as `__main__` trains on a held-out split from `url_with_headlines.csv` and saves `model.pkl`. |
| `modelBERT.py` | `Model` + `get_model()`: RoBERTa (`roberta-base`) sequence classification. Auto-loads a PyTorch checkpoint if `model.pt` or `modelBERT.pt` exists next to the script (or in the current working directory). Running as `__main__` trains and saves `modelBERT.pt`. |
| `modelNN.py` | `Model` + `get_model()`: TF-IDF (up to 30k features, bigrams) + a small MLP head. Auto-loads `model.pt` or `modelNN.pt` next to the script or in the CWD. Running as `__main__` trains and saves `model.pt`. |
| `eval_project_b.py` | Imports the model and preprocessing modules, runs `prepare_data` on a validation CSV, runs batched inference, prints `num_examples`, `avg_infer_ms`, `total_infer_s`, and `accuracy`. Optional: `--weights` (PyTorch checkpoint), `--batch-size`. |

## Data

| File | Description |
|------|-------------|
| `url_with_headlines.csv` | One row per article: **`url`**, **`headline`**. Ground-truth outlet is implied by the domain (`foxnews.com` → FoxNews, `nbcnews.com` → NBC). Preprocessing uses the given headline text (cleaned) as model input. |
| `url_only_data.csv` | **`url`** column only (same URLs as the headline dataset). Preprocessing has no headline column: it tries to **scrape** a title from each page, then falls back to a **slug-derived** string from the URL path if scraping fails or returns empty. |
| `scraped_headlines.csv` | Optional artifact: **`headline`**, **`label`** pairs produced when running `preprocess.py` as a script (writes cleaned `X`/`y` for inspection). Not required for training or `eval_project_b.py`. |
| `model.pkl` | Pickle artifact for `model.py`: TF-IDF vectorizer + `LogisticRegression` (loaded automatically when present). |
| `modelBERT.pt` | PyTorch checkpoint for `modelBERT.py` (Hugging Face classifier state + metadata). The class also accepts a generic `model.pt` in the same search order if you prefer a single filename. |
| `modelNN.pt` | PyTorch checkpoint for `modelNN.py` (vectorizer + MLP `state_dict`). The class also checks for `model.pt` first in the same directories. |
| `model.pt` | Generic PyTorch weights name still checked by `modelBERT.py` / `modelNN.py` for auto-load; `modelNN.py`’s `__main__` saves here. Use distinct files (`modelBERT.pt`, `modelNN.pt`) when both neural checkpoints live in the repo so loaders do not pick up the wrong architecture. |

**Note:** `*.pt` and `*.pkl` are listed in `.gitignore`; add trained artifacts locally or distribute them outside git as needed for submission.

## Project workflow and usage

1. **Environment** — Install dependencies from the project root:

   ```bash
   pip install -r requirements.txt
   ```

2. **Train (optional)** — Fit a classifier and write weights (each script uses `url_with_headlines.csv` and its own split):

   ```bash
   python3 model.py          # -> model.pkl (TF-IDF + logistic regression)
   python3 modelBERT.py      # -> modelBERT.pt (RoBERTa)
   python3 modelNN.py        # -> model.pt (TF-IDF + MLP; rename/copy to modelNN.pt to avoid clashing with BERT)
   ```

3. **Evaluate** — Run the course evaluator on labeled data built by your preprocessor. Examples:

   - Headlines provided in the CSV (fastest; no live HTTP):

     ```bash
     python3 eval_project_b.py --model model.py --preprocess preprocess.py --csv url_with_headlines.csv
     ```

   - URL-only CSV (scraping + slug fallback; slower and depends on network / site HTML):

     ```bash
     python3 eval_project_b.py --model model.py --preprocess preprocess.py --csv url_only_data.csv
     ```

   - BERT model with explicit checkpoint:

     ```bash
     python3 eval_project_b.py --model modelBERT.py --preprocess preprocess.py --csv url_with_headlines.csv --weights modelBERT.pt
     ```

   - TF-IDF + MLP (`modelNN.py`) with explicit checkpoint:

     ```bash
     python3 eval_project_b.py --model modelNN.py --preprocess preprocess.py --csv url_with_headlines.csv --weights modelNN.pt
     ```

   (`modelBERT.py` / `modelNN.py` also auto-load matching `model.pt` / `modelBERT.pt` / `modelNN.pt` from the project directory when you omit `--weights`, if those files exist.)

4. **Inspect preprocessing output** — To regenerate `scraped_headlines.csv` from the preprocessor’s main block, run:

   ```bash
   python3 preprocess.py
   ```
