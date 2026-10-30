# smishield

An SMS smishing and spam detector that runs **100% in your browser**. Paste a text message and you get a verdict (likely spam/smishing or likely legitimate), a confidence percentage, and a plain-English explanation of the signals behind it.

**Privacy:** the page downloads a small model file (`model.json`, about 140 KB) once. After that, every message is scored by JavaScript in your tab. Nothing you paste is sent to a server, stored, or logged.

## The problem

*Smishing* (SMS phishing) is one of the most common ways people get phished. Texts get opened far more often than email, phones show little of a link's real destination, and the messages are short enough to copy and send by the million. Typical lures are fake parcel or redelivery fees, bank or account "security alerts", unpaid tolls or taxes, and prize or refund offers. Almost all of them push you to tap a link or call a number quickly.

smishield shows how a small, explainable text classifier can flag these messages on the device without sending them anywhere.

## How it works

```
train.py (Python, run once)                     docs/ (static site, TypeScript)
───────────────────────────                     ─────────────────────────────────
SMS Spam Collection ─► tokenizer ─► TF-IDF  ─►  model.json ─► tokenizer ─► TF-IDF ─► logistic ─► verdict
(5,574 labelled SMS)   (uni+bigrams, 3,000      metrics.json  (same rules)  (same     regression   + explanation
                        terms) ─► logistic                                   idf)      (sigmoid)
                        regression
```

1. **Tokenizer** (`train.py` and `src/tokenizer.ts`, kept identical). It lowercases the text and splits it into `[a-z0-9]+` words. URLs become `__url__`, and links on known shorteners (bit.ly, tinyurl.com, …) also get `__shorturl__`. Currency symbols (£ $ €) become `__currency__`, and digit runs of 5 or more (phone numbers, short codes) become `__longnum__`. Adjacent tokens are then joined into bigrams.
2. **TF-IDF.** Uses sublinear term frequency (`1 + ln tf`), smoothed IDF and L2 normalisation, over a vocabulary capped at the 3,000 most frequent unigrams and bigrams (min document frequency 2).
3. **Logistic regression** (scikit-learn, `liblinear`, balanced class weights). `C` is chosen by 5-fold cross-validated F1 on the training split only.
4. **Export.** `train.py` writes the vocabulary, IDF values, coefficients, intercept, label mapping and decision threshold to `docs/model.json`, and the evaluation results to `docs/metrics.json`.
5. **In-browser scoring** (`src/model.ts`). Plain TypeScript reimplements the vectoriser and computes `P(spam) = sigmoid(intercept + Σ tfidfᵢ · coefᵢ)`. A message is flagged as spam when `P(spam) ≥ 0.5`, the same rule used at training time. No ML runtime is involved.
6. **Explanation** (`src/signals.ts`). The explanation lists the words and phrases that contributed most to the score (TF-IDF weight × coefficient). It also lists pattern-based warning signs: shortened URLs, other links, urgency language ("act now", "limited time", "valid 12hrs"), prize/money language, account/payment/delivery lures, premium-rate numbers and short codes, and an optional suspicious-sender check (email-to-SMS senders, short codes, look-alike names such as `AMAZ0N`). These pattern checks explain the result but **do not change the score**.

The test suite checks that the shipped artifact matches training. `tests/fixtures/parity.json` holds 100 held-out test messages together with the tokens and probabilities scikit-learn produced for them. The TypeScript code has to reproduce the same tokens and decisions, with probabilities within 1e-9.

## Measured metrics

These come from a stratified 80/20 split with seed 42: 4,459 training messages and 1,115 test messages (966 legitimate, 149 spam). The positive class is spam. Source: `docs/metrics.json`.

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9901 |
| Precision | 0.9792 |
| Recall    | 0.9463 |
| F1        | 0.9625 |

Confusion matrix on the held-out test set:

|                     | Predicted legitimate | Predicted spam |
|---------------------|---------------------:|---------------:|
| Actually legitimate | 963                  | 3              |
| Actually spam       | 8                    | 141            |

The selected regularisation was `C = 10` (cross-validated F1 0.9624), trained with scikit-learn 1.9.1. The dataset file contains 5,574 messages (4,827 ham, 747 spam). Some copies of this dataset, such as the Kaggle CSV, contain 5,572 rows.

## Caveats

- **2012-era data.** The SMS Spam Collection was collected around 2011–2012, mostly from UK and Singapore sources. Its spam is dominated by premium-rate prize, ringtone and dating texts. Modern smishing (fake parcel fees, bank and toll alerts, shortened links) looks quite different, and the model can and does miss it. The pattern checks help explain such messages but don't affect the verdict.
- **English only.** Other languages will not be scored reliably.
- **A demo and learning project.** It is **not a substitute for carrier spam filtering** or your phone's built-in protections. A "likely legitimate" result is not a guarantee that a message is safe.
- The metrics above describe this dataset's test split. They do not predict real-world performance on today's messages.

## Running the demo locally

```sh
npm install
npm run build        # tsc --noEmit typecheck + esbuild bundle → docs/app.js
npm run serve        # python3 -m http.server on http://localhost:8000
```

The page has to be served over HTTP, because opening `docs/index.html` from disk blocks `fetch("model.json")`.

To deploy on **GitHub Pages**, set the source to the `main` branch, `/docs` folder. All files the page needs (`index.html`, `style.css`, `app.js`, `model.json`, `metrics.json`) are committed in `docs/`.

## Tests

```sh
npm test             # vitest run
```

| File | Covers |
|------|--------|
| `tests/tokenizer.test.ts` | lowercasing, punctuation, URL / shortened-URL / currency / long-number handling, bigrams, config matches `model.json` |
| `tests/vectorizer.test.ts` | known unigrams and bigrams map to the right vocabulary indices, OOV handling, sublinear TF-IDF, L2 norm |
| `tests/scorer.test.ts` | real held-out spam and ham messages from the dataset classify correctly with the shipped model, confidence, sigmoid stability |
| `tests/parity.test.ts` | 100 held-out messages: TypeScript tokens, probabilities and decisions match scikit-learn |
| `tests/signals.test.ts` | a message with a shortened URL and urgency words surfaces both signals, plus prize/money, premium numbers, account lures, sender checks and summary wording |

All tests load `docs/model.json`, the same file the web page uses.

## Retraining

Python is only used for this one-time training step. The virtual environment and the downloaded dataset are gitignored.

```sh
python3 -m venv .venv
.venv/bin/pip install scikit-learn
.venv/bin/python train.py   # or: source .venv/bin/activate && python3 train.py
npm test && npm run build
```

`train.py` first tries the original UCI URL (`…/machine-learning-databases/00228/SMSSpamCollection.zip`). As of this writing that URL returns 404, so the script falls back to UCI's current static URL and then to GitHub mirrors of the same file. The download is cached in `data/` (gitignored). The script prints the metrics and confusion matrix, then rewrites `docs/model.json`, `docs/metrics.json` and `tests/fixtures/parity.json`. If you change the tokenizer in `train.py`, make the same change in `src/tokenizer.ts`. The parity test fails if the two differ.

## Project layout

```
train.py                 one-time training + export (Python, scikit-learn)
src/tokenizer.ts         tokenizer, mirrored from train.py
src/model.ts             model loading, TF-IDF vectoriser, sigmoid scorer
src/signals.ts           pattern-based warning signs + plain-English explanation
src/main.ts              browser UI
docs/                    static site (GitHub Pages): index.html, style.css, app.js, model.json, metrics.json
tests/                   vitest suites + parity fixture
```

## Dataset and credits

T. A. Almeida, J. M. Gómez Hidalgo and A. Yamakami, *"Contributions to the Study of SMS Spam Filtering: New Collection and Results"*, ACM DocEng 2011. The SMS Spam Collection is published in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (CC BY 4.0). The raw dataset is not redistributed here, apart from the 100 test messages in the parity fixture.
