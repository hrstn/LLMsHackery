# IMDB Sentiment Analysis — HTB Skills Assessment

> **One job:** read a movie review, say if it's positive (1) or negative (0).

---

## TL;DR — just run this

```bash
pip install scikit-learn requests joblib scipy
python skills_assessment.py <TARGET_IP>
```

That's it. The script trains the model, saves it, and uploads it automatically.

---

## What's in the folder

```
IMDB Sentiment/
├── skills_assessment.py        ← the script you run
├── skills_assessment.joblib    ← the trained model (already generated)
├── imdb_data/
│   ├── train.json              ← 25,000 reviews for training
│   └── test.json               ← 25,000 reviews for testing
└── requirements.txt
```

---

## The data

Two JSON files. Each review looks like this:

```json
{ "text": "This movie was fantastic...", "label": 1 }
{ "text": "Worst film I've ever seen.", "label": 0 }
```

- `label: 1` = positive review
- `label: 0` = negative review
- 25,000 reviews each side — perfectly balanced

---

## What the script does, step by step

### Step 1 — Download (skipped if data already exists)

```python
if not os.path.exists("imdb_data/train.json"):
    download_and_extract(url)
```

Grabs a ZIP from the HTB server, unpacks it into `imdb_data/`.
Already have it? This step does nothing.

---

### Step 2 — Load the data

```python
with open("imdb_data/train.json", encoding="utf-8") as f:
    train_data = json.load(f)

X_train = [s["text"] for s in train_data]   # list of review strings
y_train = [s["label"] for s in train_data]  # list of 0s and 1s
```

`X_train` = the words.
`y_train` = the answers.

Same thing for `test.json` → `X_test`, `y_test`.

---

### Step 3 — Build the pipeline

```python
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
```

A **Pipeline** chains two steps. Give it raw text, it handles everything internally.

**Step 1 inside — TfidfVectorizer:**
Converts text into numbers. Each word gets a score based on how often it appears
in this review vs. how common it is across all reviews.

- "fantastic" appearing only in positive reviews → high score when present
- "the" appearing everywhere → near-zero score

The result is a giant row of numbers, one per unique word in the vocabulary.

**Step 2 inside — LogisticRegression:**
Takes those numbers and learns which combinations point to positive vs negative.
After training: if the number for "brilliant" goes up, lean toward 1.
If the number for "terrible" goes up, lean toward 0.

---

### Step 4 — Train

```python
pipeline.fit(X_train, y_train)
```

One line. The pipeline:
1. Builds the vocabulary from all 25,000 training reviews
2. Converts every review into a number vector
3. Trains the classifier on those vectors + labels

Takes about 30 seconds.

---

### Step 5 — Evaluate

```python
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

Runs 25,000 unseen reviews through the trained model.
Compares predictions to the real labels.

Expected output:
```
Accuracy: 0.88292
```

88% correct on reviews it has never seen. Good enough.

---

### Step 6 — The compatibility fix (important)

```python
tfidf_trans = pipeline.named_steps["vectorizer"]._tfidf
if not hasattr(tfidf_trans, "_idf_diag"):
    idf = tfidf_trans.idf_
    n   = len(idf)
    tfidf_trans._idf_diag = sp.diags(idf, 0, shape=(n, n), format="csr", dtype=idf.dtype)
```

**Why this exists:**

The HTB server runs **sklearn 1.6**. Your machine probably runs something newer.

In sklearn 1.6, the vectorizer stores the word weights as a sparse diagonal matrix
called `_idf_diag`. In sklearn 1.8+, they removed it and do the same thing differently.

If you skip this step, the server loads your model, calls `predict()`, hits a missing
attribute, crashes silently, and returns `{"accuracy": 0.0}`. No error message. Just
zero. Very annoying to debug.

This line manually adds `_idf_diag` back so the old server can still use the model.

---

### Step 7 — Save

```python
joblib.dump(pipeline, "skills_assessment.joblib", compress=False, protocol=4)
```

Serialises the entire trained pipeline to disk.

- `compress=False` — no compression, simpler format, fewer compatibility issues
- `protocol=4` — Python 3.4+ readable; without this, Python 3.14 writes protocol 5
  which older Python versions cannot open

The file is ~3.4 MB.

---

### Step 8 — Upload

```python
url = f"http://{target_ip}:5000/api/upload"
with open("skills_assessment.joblib", "rb") as f:
    r = requests.post(url, files={"model": f}, timeout=60)
print(r.json())
```

Sends the file to the HTB evaluation server.

Server loads it, runs it against its own test set, returns:
```json
{
    "accuracy": 1.0,
    "flag": "HTB{...}"
}
```

---

## How predict() works after training

```python
pipeline.predict(["This film was absolutely incredible"])
# → [1]

pipeline.predict(["Boring, slow, I fell asleep"])
# → [0]
```

You can call it on any raw text string. The pipeline handles vectorisation
internally — you never touch the numbers yourself.

---

## Why TF-IDF + Logistic Regression

**TF-IDF** — counts words, but smarter than raw counts.
Words that appear in every review (like "the") get penalised.
Words that appear only in positive or negative reviews get boosted.

**Logistic Regression** — fast, interpretable, works well on sparse text data.
Learns a weight per word. Positive weight → pushes toward 1.
Negative weight → pushes toward 0.

No neural network needed. This problem is solved by counting words.

---

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `{"accuracy": 0.0, "metrics": null}` | sklearn version mismatch — `_idf_diag` missing | The fix in Step 6 |
| `{"accuracy": 0.0, "metrics": null}` | uploaded old/wrong joblib file | Re-run the script, don't upload manually |
| Download fails | HTB VPN not connected | Connect to HTB VPN first |
| `ModuleNotFoundError` | missing dependency | `pip install scikit-learn requests joblib scipy` |

---

## Requirements

```
scikit-learn >= 1.0
requests >= 2.28
joblib >= 1.2
scipy >= 0.7
```
