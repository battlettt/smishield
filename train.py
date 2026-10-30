#!/usr/bin/env python3
"""Train the smishield model and export it for the in-browser TypeScript demo.

Pipeline:
  1. Download the SMS Spam Collection (UCI archive, with fallbacks).
  2. Stratified 80/20 train/test split with a fixed seed.
  3. TF-IDF over word unigrams + bigrams (capped vocabulary) + logistic regression.
  4. Print precision / recall / F1 / accuracy and the confusion matrix.
  5. Export docs/model.json, docs/metrics.json and tests/fixtures/parity.json.

The tokenizer below is mirrored exactly by src/tokenizer.ts. If you change one,
change the other; tests/parity.test.ts will fail if they drift apart.

Usage:
  python3 -m venv .venv && .venv/bin/pip install scikit-learn
  .venv/bin/python train.py
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"  # gitignored; the raw dataset is never shipped
DOCS_DIR = ROOT / "docs"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "parity.json"

SEED = 42
TEST_SIZE = 0.2
MAX_FEATURES = 3000
MIN_DF = 2
THRESHOLD = 0.5
C_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]

SOURCES = [
    # Original URL from the dataset's UCI page.
    ("zip", "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/SMSSpamCollection.zip"),
    # Current UCI static URL for the same archive.
    ("zip", "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"),
    # GitHub mirrors of the same tab-separated file (label<TAB>message).
    ("tsv", "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"),
    ("tsv", "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"),
]

# ---------------------------------------------------------------------------
# Tokenizer (mirrored in src/tokenizer.ts)
# ---------------------------------------------------------------------------

URL_TOKEN = "__url__"
SHORT_URL_TOKEN = "__shorturl__"
CURRENCY_TOKEN = "__currency__"
LONG_NUMBER_TOKEN = "__longnum__"
LONG_NUMBER_MIN_DIGITS = 5

SHORTENER_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "bl.ink",
    "t.ly", "s.id", "v.gd", "short.io", "tiny.one", "lnkd.in", "qr.ae",
]

# Explicit character classes (no \s, \b, \d) so Python and JavaScript agree.
_NON_SPACE = r"[^ \t\n\r\f\v]"
_TLDS = r"(?:com|net|org|info|biz|co|uk|ly|gl|gd|cc|gy|at|io|me|link|ink|id|in|ae|one|us|xyz|top|ru|cn)"
URL_PATTERN = (
    r"(?:https?://|www\.)" + _NON_SPACE + r"+"
    r"|(?<![a-z0-9.-])(?:[a-z0-9-]+\.)+" + _TLDS + r"(?![a-z0-9-])(?:/" + _NON_SPACE + r"*)?"
)
TOKEN_RE = re.compile("(" + URL_PATTERN + r")|([£$€])|([a-z0-9]+)")
_DIGITS_RE = re.compile(r"^[0-9]+$")


def url_domain(url: str) -> str:
    d = re.sub(r"^https?://", "", url)
    d = re.sub(r"^www\.", "", d)
    d = d.split("/")[0].split("?")[0]
    return re.sub(r"[^a-z0-9]+$", "", d)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in TOKEN_RE.finditer(text.lower()):
        url, cur, word = m.group(1), m.group(2), m.group(3)
        if url is not None:
            tokens.append(URL_TOKEN)
            if url_domain(url) in SHORTENER_DOMAINS:
                tokens.append(SHORT_URL_TOKEN)
        elif cur is not None:
            tokens.append(CURRENCY_TOKEN)
        elif _DIGITS_RE.match(word) and len(word) >= LONG_NUMBER_MIN_DIGITS:
            tokens.append(LONG_NUMBER_TOKEN)
        else:
            tokens.append(word)
    return tokens


def analyze(text: str) -> list[str]:
    """Unigrams followed by bigrams (space-joined), like sklearn's ngram_range=(1, 2)."""
    toks = tokenize(text)
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "smishield-train/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def parse_tsv(raw: bytes) -> list[tuple[str, str]]:
    rows = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        label, _, message = line.partition("\t")
        if label not in ("ham", "spam"):
            raise ValueError(f"unexpected label {label!r}")
        rows.append((label, message))
    return rows


def load_dataset() -> tuple[list[tuple[str, str]], str]:
    DATA_DIR.mkdir(exist_ok=True)
    cached = DATA_DIR / "SMSSpamCollection"
    source_file = DATA_DIR / "SOURCE"
    if cached.exists() and source_file.exists():
        return parse_tsv(cached.read_bytes()), source_file.read_text().strip()

    errors = []
    for kind, url in SOURCES:
        try:
            print(f"Downloading {url} ...")
            raw = fetch(url)
            if kind == "zip":
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    raw = zf.read("SMSSpamCollection")
            rows = parse_tsv(raw)
            if len(rows) < 5000:
                raise ValueError(f"only {len(rows)} rows parsed")
            cached.write_bytes(raw)
            source_file.write_text(url + "\n")
            return rows, url
        except Exception as exc:  # try the next mirror
            print(f"  failed: {exc}")
            errors.append(f"{url}: {exc}")
    sys.exit("Could not download the dataset from any source:\n" + "\n".join(errors))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main() -> None:
    rows, source = load_dataset()
    texts = [m for _, m in rows]
    y = np.array([1 if label == "spam" else 0 for label, _ in rows])
    n_spam, n_ham = int(y.sum()), int(len(y) - y.sum())
    print(f"Loaded {len(rows)} messages from {source} ({n_ham} ham, {n_spam} spam)")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=analyze,
            lowercase=False,  # the analyzer lowercases itself
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            sublinear_tf=True,
            smooth_idf=True,
            norm="l2",
        )),
        ("clf", LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=1000, random_state=SEED
        )),
    ])
    # Pick the regularisation strength by cross-validated F1 on the training split only.
    search = GridSearchCV(
        pipeline,
        {"clf__C": C_GRID},
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    best_c = float(search.best_params_["clf__C"])
    print(f"Best C by 5-fold CV F1: {best_c} (cv f1={search.best_score_:.4f})")

    tfidf: TfidfVectorizer = model.named_steps["tfidf"]
    clf: LogisticRegression = model.named_steps["clf"]

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    print("\nHeld-out test set metrics (positive class = spam):")
    print(f"  accuracy : {acc:.4f}")
    print(f"  precision: {prec:.4f}")
    print(f"  recall   : {rec:.4f}")
    print(f"  f1       : {f1:.4f}")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(f"               pred ham  pred spam")
    print(f"  actual ham   {cm[0, 0]:8d}  {cm[0, 1]:9d}")
    print(f"  actual spam  {cm[1, 0]:8d}  {cm[1, 1]:9d}")

    vocab_items = sorted(tfidf.vocabulary_.items(), key=lambda kv: kv[1])
    vocabulary = [term for term, _ in vocab_items]
    assert [i for _, i in vocab_items] == list(range(len(vocabulary)))

    model_json = {
        "format": "smishield-tfidf-logreg",
        "version": 1,
        "labels": {"0": "ham", "1": "spam"},
        "positiveLabel": "spam",
        "threshold": THRESHOLD,
        "tokenizer": {
            "lowercase": True,
            "ngramRange": [1, 2],
            "urlToken": URL_TOKEN,
            "shortUrlToken": SHORT_URL_TOKEN,
            "currencyToken": CURRENCY_TOKEN,
            "longNumberToken": LONG_NUMBER_TOKEN,
            "longNumberMinDigits": LONG_NUMBER_MIN_DIGITS,
            "shortenerDomains": SHORTENER_DOMAINS,
        },
        "tfidf": {"sublinearTf": True, "smoothIdf": True, "norm": "l2"},
        "vocabulary": vocabulary,
        "idf": [float(v) for v in tfidf.idf_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
    }

    metrics_json = {
        "dataset": {
            "name": "SMS Spam Collection v.1 (Almeida & Gómez Hidalgo)",
            "source": source,
            "messages": len(rows),
            "ham": n_ham,
            "spam": n_spam,
        },
        "split": {
            "testSize": TEST_SIZE,
            "stratified": True,
            "seed": SEED,
            "train": len(X_train),
            "test": len(X_test),
            "testHam": int((y_test == 0).sum()),
            "testSpam": int((y_test == 1).sum()),
        },
        "model": {
            "features": "TF-IDF word unigrams + bigrams (sublinear tf, l2 norm)",
            "vocabularySize": len(vocabulary),
            "maxFeatures": MAX_FEATURES,
            "minDf": MIN_DF,
            "classifier": "LogisticRegression (liblinear, class_weight=balanced)",
            "C": best_c,
            "cvF1": round(float(search.best_score_), 4),
            "threshold": THRESHOLD,
            "sklearnVersion": sklearn.__version__,
        },
        "test": {
            "positiveClass": "spam",
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "confusionMatrix": {
                "labels": ["ham", "spam"],
                "rowsAreActual": True,
                "matrix": cm.tolist(),
                "trueNegatives": int(cm[0, 0]),
                "falsePositives": int(cm[0, 1]),
                "falseNegatives": int(cm[1, 0]),
                "truePositives": int(cm[1, 1]),
            },
        },
    }

    # Parity fixture: held-out messages with Python's tokens and probabilities, so the
    # TypeScript test suite can prove the shipped model.json reproduces training-time output.
    rng = np.random.default_rng(SEED)
    spam_idx = [i for i in range(len(X_test)) if y_test[i] == 1]
    ham_idx = [i for i in range(len(X_test)) if y_test[i] == 0]
    picked = sorted(
        list(rng.choice(spam_idx, 40, replace=False)) + list(rng.choice(ham_idx, 60, replace=False))
    )
    fixture = {
        "note": "Held-out test messages from the SMS Spam Collection with training-time outputs.",
        "threshold": THRESHOLD,
        "cases": [
            {
                "text": X_test[i],
                "label": "spam" if y_test[i] == 1 else "ham",
                "tokens": tokenize(X_test[i]),
                "probability": float(proba[i]),
                "predicted": "spam" if pred[i] == 1 else "ham",
            }
            for i in picked
        ],
    }

    DOCS_DIR.mkdir(exist_ok=True)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "model.json").write_text(json.dumps(model_json, separators=(",", ":"), ensure_ascii=False))
    (DOCS_DIR / "metrics.json").write_text(json.dumps(metrics_json, indent=2, ensure_ascii=False) + "\n")
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {DOCS_DIR / 'model.json'} ({len(vocabulary)} features)")
    print(f"Wrote {DOCS_DIR / 'metrics.json'}")
    print(f"Wrote {FIXTURE_PATH} ({len(fixture['cases'])} cases)")


if __name__ == "__main__":
    main()
