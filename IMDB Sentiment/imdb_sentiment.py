import requests
import zipfile
import tarfile
import io
import os
import re
import json
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


MODEL_FILE = "imdb_sentiment_model.joblib"
DATA_DIR = "imdb_data"

HTB_URL = "https://academy.hackthebox.com/storage/modules/290/imdb_sentiment_dataset.zip"


def download_htb(url=HTB_URL):
    print(f"Downloading from {url} ...")
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f"HTTP {r.status_code} — download failed")
        return False
    os.makedirs(DATA_DIR, exist_ok=True)
    if url.endswith(".tar.gz") or url.endswith(".tgz"):
        with tarfile.open(fileobj=io.BytesIO(r.content)) as t:
            t.extractall(DATA_DIR)
    else:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(DATA_DIR)
            print("Contents:", z.namelist())
    return True


def load_imdb_nltk():
    """Fallback: load IMDB reviews via NLTK's movie_reviews corpus."""
    import nltk
    nltk.download("movie_reviews", quiet=True)
    from nltk.corpus import movie_reviews
    docs = [(movie_reviews.raw(fid), cat)
            for cat in movie_reviews.categories()
            for fid in movie_reviews.fileids(cat)]
    texts, labels = zip(*docs)
    df = pd.DataFrame({"text": texts, "label": [1 if l == "pos" else 0 for l in labels]})
    print(f"Loaded {len(df)} reviews from NLTK movie_reviews corpus")
    return df


def load_splits():
    """Load train and test splits from DATA_DIR, preserving the dataset's original split."""
    train_path = os.path.join(DATA_DIR, "train.json")
    test_path = os.path.join(DATA_DIR, "test.json")

    if os.path.exists(train_path) and os.path.exists(test_path):
        df_train = pd.read_json(train_path, orient="records")
        df_test = pd.read_json(test_path, orient="records")
        print(f"Loaded train ({len(df_train)} rows) and test ({len(df_test)} rows)")
        return df_train, df_test

    # Single-file fallback: sort to get deterministic order, split 80/20
    for root, _, files in os.walk(DATA_DIR):
        for fname in sorted(files):
            path = os.path.join(root, fname)
            if fname.endswith(".json"):
                df = pd.read_json(path, orient="records")
            elif fname.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                continue
            print(f"Loaded {path} ({len(df)} rows) — splitting 80/20")
            return train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    print("No dataset found in imdb_data/ — using NLTK movie_reviews corpus")
    df = load_imdb_nltk()
    return train_test_split(df, test_size=0.2, random_state=42, shuffle=True)


def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)        # HTML tags
    text = re.sub(r"&\w+;", " ", text)        # HTML entities e.g. &amp;
    text = re.sub(r"[^\w\s]", " ", text)      # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def preprocess(df):
    rename = {}
    for c in df.columns:
        if c.lower() in ("review", "text", "comment", "content"):
            rename[c] = "text"
        elif c.lower() in ("sentiment", "label", "class", "target"):
            rename[c] = "label"
    df = df.rename(columns=rename)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected 'text' and 'label' columns; got {df.columns.tolist()}")

    if df["label"].dtype == object:
        df["label"] = df["label"].map(
            lambda x: 1 if str(x).strip().lower() in ("positive", "pos", "1") else 0
        )

    df = df.drop_duplicates().reset_index(drop=True)
    df["text"] = df["text"].astype(str).apply(clean_text)
    pos = int(df["label"].sum())
    print(f"  {len(df)} reviews — {pos} positive, {len(df)-pos} negative")
    return df


def train_model(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    print("Training ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    return pipeline


def save_model(model, path=MODEL_FILE):
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def upload_model(target_ip, path=MODEL_FILE):
    url = f"http://{target_ip}:5000/api/upload"
    print(f"Uploading to {url} ...")
    with open(path, "rb") as f:
        r = requests.post(url, files={"model": f}, timeout=30)
    print(json.dumps(r.json(), indent=4))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <target_ip>  [dataset_url]")
        sys.exit(1)

    target = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else HTB_URL

    download_htb(url)
    df_train, df_test = load_splits()

    print("Preprocessing train set:")
    df_train = preprocess(df_train)
    print("Preprocessing test set:")
    df_test = preprocess(df_test)

    model = train_model(
        df_train["text"], df_test["text"],
        df_train["label"], df_test["label"],
    )
    save_model(model)
    upload_model(target)
