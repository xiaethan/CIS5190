import re
from typing import List, Tuple, Optional
from urllib.parse import urlparse, unquote

import pandas as pd


def clean_text(text: str) -> str:
    """
    This function cleans each headline.
    It removes HTML tags, fixes curly quotation marks, removes suffixes,
    and removes extra spaces.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize quotation marks using Unicode codes
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Remove common source suffixes from page titles
    # Handles: " - Fox News", " | Fox News", " – Fox News", " — Fox News"
    text = re.sub(r"\s*[-\|\u2013\u2014]\s*Fox News\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[-\|\u2013\u2014]\s*NBC News\s*$", "", text, flags=re.IGNORECASE)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Find a column name in a flexible way.
    For example, this treats 'URL', 'url', and 'Url' similarly.
    """
    normalized_cols = {
        col.lower().strip().replace(" ", "_"): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.lower().strip().replace(" ", "_")
        if key in normalized_cols:
            return normalized_cols[key]

    return None


def label_from_url(url: str) -> str:
    """
    Create the ground-truth label from the URL.
    The model will not receive this label; it is only y for evaluation.
    """
    url_lower = str(url).lower()

    if "foxnews.com" in url_lower:
        return "FoxNews"
    elif "nbcnews.com" in url_lower:
        return "NBC"
    else:
        raise ValueError(f"Cannot infer label from URL: {url}")


def headline_from_url_slug(url: str) -> str:
    """
    Backup method if the CSV has no headline and scraping fails.

    Example:
    https://www.foxnews.com/politics/biden-meets-leaders
    becomes:
    biden meets leaders
    """
    parsed = urlparse(str(url))
    path = parsed.path.strip("/")

    if not path:
        return ""

    slug = path.split("/")[-1]
    slug = unquote(slug)

    # Remove file endings
    slug = re.sub(r"\.(html|htm|print)$", "", slug, flags=re.IGNORECASE)

    # Remove NBC article IDs at the end, such as rcna166855 or ncna1298934
    slug = re.sub(r"[-_]*(rcna|ncna)\d+$", "", slug, flags=re.IGNORECASE)

    # Convert URL style words into readable text
    text = slug.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def scrape_headline(url: str) -> str:
    """
    Try to scrape the headline from a Fox/NBC article URL.
    If anything fails, return an empty string and use the URL slug instead.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=8)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # Usually the best source is metadata
        for attr, value in [
            ("property", "og:title"),
            ("name", "twitter:title"),
            ("name", "title"),
        ]:
            tag = soup.find("meta", attrs={attr: value})
            if tag is not None and tag.get("content"):
                return clean_text(tag["content"])

        # Backup: find the first h1
        h1 = soup.find("h1")
        if h1 is not None:
            return clean_text(h1.get_text(" ", strip=True))

        return ""

    except Exception:
        return ""


def prepare_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Required by the evaluator.

    Args:
        csv_path: path to csv file, such as url_with_headlines.csv or url_only_data.csv

    Returns:
        X: list of headline strings
        y: list of labels, either "FoxNews" or "NBC"
    """

    # 1. Read CSV
    df = pd.read_csv(csv_path)

    # 2. Find URL column
    url_col = find_column(df, ["url", "link", "article_url"])

    if url_col is None:
        raise ValueError("CSV must contain a URL column, such as 'url'.")

    # 3. Find headline column if it exists
    headline_col = find_column(df, ["headline", "title", "article_title"])

    X = []
    y = []

    for _, row in df.iterrows():
        url = str(row[url_col])

        # Ground-truth label
        label = label_from_url(url)

        # Case 1: use headline column if the CSV already has it
        headline = ""
        if headline_col is not None and not pd.isna(row[headline_col]):
            headline = clean_text(row[headline_col])

        # Case 2: if no headline is provided, try scraping
        if headline == "":
            headline = scrape_headline(url)

        # Case 3: if scraping fails, use URL slug as backup
        if headline == "":
            headline = clean_text(headline_from_url_slug(url))

        # Keep valid examples only
        if headline != "":
            X.append(headline)
            y.append(label)

    return X, y


# This part is just for checking the format of X and y
if __name__ == "__main__":
    X, y = prepare_data("url_with_headlines.csv")

    print("Number of examples:", len(X))
    print("Number of labels:", len(y))
    print("Unique labels:", set(y))

    print("\nFirst 5 examples:")
    for i in range(5):
        print("X:", X[i])
        print("y:", y[i])
        print("-" * 50)