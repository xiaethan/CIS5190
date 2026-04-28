# CIS5190

**CIS5190 Final Project**

- **Project members:** Yixi Tan, Georges Chebly, Ethan Xia
- **Project type:** News Source Classification

**Final project report:** [Overleaf](https://www.overleaf.com/8817778222wsgtvdjzmjhq#8f728f)

This project builds a classifier that predicts which news outlet produced a story (**FoxNews** vs **NBC**) from headline text. Labels are derived from the article URL domain during preprocessing; the model only sees text features derived from headlines (either provided in the CSV or obtained by scraping or URL-slug fallback).

## Scripts

| File | Description |
|------|-------------|
| `preprocess.py` | Implements `prepare_data(csv_path)` for the evaluator: reads a CSV with a `url` column and optional `headline` / `title` column, builds aligned inputs `X` (cleaned headline strings) and labels `y` (`FoxNews` or `NBC`). If no headline is present, fetches the page with HTTP and parses metadata/HTML (`requests` + BeautifulSoup); if that fails, falls back to text derived from the URL path. |
| `model.py` | `Model` class (`nn.Module`) with `predict(batch)` and `get_model()`: TF-IDF features + scikit-learn `LogisticRegression`, trained weights in `model.pkl` when present. Running as `__main__` trains on a held-out split from `url_with_headlines.csv` and saves `model.pkl`. |
| `eval_project_b.py` | Imports the model and preprocessing modules, runs `prepare_data` on a validation CSV, runs batched inference, prints `num_examples`, `avg_infer_ms`, `total_infer_s`, and `accuracy`. Optional: `--weights` (PyTorch checkpoint), `--batch-size`. |

## Data

| File | Description |
|------|-------------|
| `url_with_headlines.csv` | One row per article: **`url`**, **`headline`**. Ground-truth outlet is implied by the domain (`foxnews.com` → FoxNews, `nbcnews.com` → NBC). Preprocessing uses the given headline text (cleaned) as model input. |
| `url_only_data.csv` | **`url`** column only (same URLs as the headline dataset). Preprocessing has no headline column: it tries to **scrape** a title from each page, then falls back to a **slug-derived** string from the URL path if scraping fails or returns empty. |
| `scraped_headlines.csv` | Optional artifact: **`headline`**, **`label`** pairs produced when running `preprocess.py` as a script (writes cleaned `X`/`y` for inspection). Not required for training or `eval_project_b.py`. |
| `model.pkl` | Serialized TF-IDF vectorizer + logistic regression weights produced by training in `model.py` (loaded automatically by `Model` / `get_model()` if the file exists). |

## Project workflow and usage

1. **Environment** — Install dependencies from the project root:

   ```bash
   pip install -r requirements.txt
   ```

2. **Train (optional)** — Fit the classifier and write `model.pkl` (uses `url_with_headlines.csv` and a train/test split inside `model.py`):

   ```bash
   python3 model.py
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

4. **Inspect preprocessing output** — To regenerate `scraped_headlines.csv` from the preprocessor’s main block, run:

   ```bash
   python3 preprocess.py
   ```

## Project Results

Benchmarks from local `eval_project_b.py` runs:

- **No headline in input CSV** (`url_only_data.csv`)
  - `num_examples`: 3805
  - `avg_infer_ms`: 0.020
  - `total_infer_s`: 0.076
  - `accuracy`: 0.960578

- **With headline in input CSV** (`url_with_headlines.csv`)
  - `num_examples`: 3805
  - `avg_infer_ms`: 0.018
  - `total_infer_s`: 0.069
  - `accuracy`: 0.960578
