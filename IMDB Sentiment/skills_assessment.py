#!/usr/bin/env python3
"""
HTB Skills Assessment — IMDB Sentiment Analysis
Usage: python skills_assessment.py <target_ip> [dataset_url]
"""

import io
import json
import os
import sys
import zipfile
import tarfile

import joblib
import requests
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

MODEL_FILE = "skills_assessment.joblib"
DATA_DIR   = "imdb_data"
HTB_URL    = "https://academy.hackthebox.com/storage/modules/290/imdb_sentiment_dataset.zip"


# ── Download ──────────────────────────────────────────────────────────────────

def download_and_extract(url: str = HTB_URL, dest: str = DATA_DIR) -> bool:
    print(f"[download] {url}")
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
    except Exception as exc:
        print(f"[download] FAILED: {exc}")
        return False
    os.makedirs(dest, exist_ok=True)
    raw = r.content
    if ".tar" in url or url.endswith(".tgz"):
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            t.extractall(dest)
    else:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            print(f"[download] contents: {z.namelist()}")
            z.extractall(dest)
    print(f"[download] OK ({len(raw)/1e6:.1f} MB)")
    return True


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(data_dir: str = DATA_DIR):
    train_p = os.path.join(data_dir, "train.json")
    test_p  = os.path.join(data_dir, "test.json")
    with open(train_p, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_p, encoding="utf-8") as f:
        test_data = json.load(f)
    X_train = [s["text"] for s in train_data]
    y_train = [s["label"] for s in train_data]
    X_test  = [s["text"] for s in test_data]
    y_test  = [s["label"] for s in test_data]
    print(f"[load] train={len(X_train)}  test={len(X_test)}")
    return X_train, y_train, X_test, y_test


# ── Train ─────────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_test, y_test) -> Pipeline:
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    print("[train] fitting ...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"[eval]  accuracy = {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    return pipeline


# ── Export ────────────────────────────────────────────────────────────────────

def export(pipeline: Pipeline, path: str = MODEL_FILE) -> str:
    # sklearn <=1.6 TfidfTransformer.transform() requires _idf_diag (a sparse
    # CSR diagonal IDF matrix).  sklearn >=1.8 removed this attribute and now
    # uses idf_ directly.  Inject _idf_diag so the server's older sklearn can
    # still call predict() without AttributeError.
    tfidf_trans = pipeline.named_steps["vectorizer"]._tfidf
    if not hasattr(tfidf_trans, "_idf_diag"):
        idf = tfidf_trans.idf_
        n   = len(idf)
        tfidf_trans._idf_diag = sp.diags(
            idf, offsets=0, shape=(n, n), format="csr", dtype=idf.dtype
        )
        print(f"[export] injected _idf_diag ({n}x{n} csr) for sklearn <=1.6")

    # protocol=4: Python 3.4+ readable (avoids Python 3.8-only protocol 5
    # that our Python 3.14 would write by default).
    joblib.dump(pipeline, path, compress=False, protocol=4)
    print(f"[export] saved → {path}  ({os.path.getsize(path)/1e3:.0f} KB)")

    # Sanity reload
    loaded = joblib.load(path)
    preds  = loaded.predict([
        "This film was absolutely fantastic.",
        "Terrible, boring, a complete waste of time.",
    ])
    assert list(map(int, preds)) == [1, 0], f"Sanity check FAILED: {preds}"
    print(f"[export] reload OK — predict={preds}")
    return path


# ── Upload ────────────────────────────────────────────────────────────────────

def upload(target_ip: str, model_path: str = MODEL_FILE) -> None:
    url = f"http://{target_ip}:5000/api/upload"
    print(f"\n[upload] POST {url}")
    with open(model_path, "rb") as f:
        r = requests.post(url, files={"model": f}, timeout=60)
    print(f"[upload] HTTP {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=4))
    except Exception:
        print(r.text)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <target_ip> [dataset_url]")
        sys.exit(1)

    target = sys.argv[1]
    url    = sys.argv[2] if len(sys.argv) > 2 else HTB_URL

    if not os.path.exists(os.path.join(DATA_DIR, "train.json")):
        if not download_and_extract(url):
            sys.exit("[error] download failed")

    X_train, y_train, X_test, y_test = load_data()
    pipeline = train(X_train, y_train, X_test, y_test)
    export(pipeline)
    upload(target)
