# Emotion Classification with RNN, LSTM, GRU & BiGRU

Classifying short texts into six emotions (**sadness, joy, love, anger, fear, surprise**) using recurrent neural networks, compared against a classical TF-IDF + Logistic Regression baseline.

The best recurrent models reach **~92% test accuracy and ~0.88 macro-F1**, about 5 points above the baseline.

---

## Highlights

- Compared **Simple RNN, LSTM, GRU and stacked Bidirectional GRU** under identical training conditions
- Added a **TF-IDF + Logistic Regression baseline** to check whether the deep models are worth the complexity
- Handled class imbalance with **balanced class weights**
- Used **masking** so recurrent layers ignore padding tokens
- Selected models on the **validation split**; the **test split is used only for final reporting**
- Per-class precision/recall/F1, raw and normalized confusion matrices, and manual error analysis

---

## Dataset

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) from Hugging Face: English Twitter messages labeled with one of six emotions.

| Split | Samples |
|---|---|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

The classes are imbalanced: joy (33.5%) and sadness (29.2%) dominate, while surprise is only 3.6%. The labels were assigned automatically from hashtags (distant supervision), so some labels are noisy or ambiguous. This limits the achievable accuracy.

---

## Approach

**Preprocessing**
- Keras `Tokenizer` (vocabulary capped at 10,000 words, `<unk>` for out-of-vocabulary), fit on the training set only
- Sequences padded/truncated to 50 tokens (covers 98.7% of sentences without truncation)
- `mask_zero=True` in the embedding layer so padding doesn't affect the recurrent state

**Models**

| Model | Architecture |
|---|---|
| Simple RNN | Embedding(128) → SimpleRNN(128) → Dropout(0.5) → SimpleRNN(64) → Dropout(0.5) → Dense(6) |
| LSTM | Same layout with LSTM layers |
| GRU | Same layout with GRU layers |
| BiGRU | Embedding(300) → BiGRU(128) → Dropout(0.5) → BiGRU(64) → Dropout(0.5) → Dense(6) |
| Baseline | TF-IDF (1-2 grams) → Logistic Regression (balanced class weights) |

**Training:** Adam optimizer, sparse categorical cross-entropy, batch size 32, up to 20 epochs, early stopping on validation loss (patience 3, best weights restored), balanced class weights.

---

## Results

| Model | Val Accuracy | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| BiGRU | 92.6% | 91.75% | 0.878 |
| LSTM | 91.75% | 91.65% | 0.879 |
| GRU | 92.1% | 91.05% | 0.877 |
| TF-IDF + LogReg | 87.2% | 86.85% | 0.834 |
| Simple RNN | 82.0% | 82.6% | 0.756 |

*Exact numbers vary slightly between runs.*

**Per-class results for BiGRU (test set)**

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| sadness | 0.975 | 0.940 | 0.957 |
| joy | 0.962 | 0.914 | 0.937 |
| love | 0.767 | 0.912 | 0.833 |
| anger | 0.915 | 0.945 | 0.930 |
| fear | 0.934 | 0.821 | 0.874 |
| surprise | 0.591 | 0.985 | 0.739 |

![Confusion matrices](images/confusion_matrix.png)

### Key observations

- **Gated models clearly beat the baseline and the plain RNN.** LSTM, GRU and BiGRU all land around 91-92%, roughly 5 points above TF-IDF + Logistic Regression and about 9 points above the Simple RNN, whose training was also unstable.
- **BiGRU did not meaningfully outperform LSTM or GRU.** With 2,000 test samples, differences under about 1 point are within noise. Bidirectionality and a larger embedding added complexity without a clear gain on short sentences.
- **The model over-predicts the minority classes.** Surprise has very high recall (0.985) but low precision (0.59), and love shows a milder version of the same pattern. This is likely a side effect of the large class weights (surprise is weighted about 4.7x), and it is what holds macro-F1 down.
- **Love and joy are the most commonly confused pair**, which makes sense because they overlap semantically.
- **Many errors are ambiguous or noisy labels.** For example, near-identical phrases such as "feeling generous" appear under different labels, so the practical ceiling is in the low 90s.

---

## Limitations

- Training data is mostly short, first-person "i feel ..." statements, so the model may generalize less well to other writing styles. For example, an enthusiastic sentence like "I can't believe how happy I am right now, this is amazing!" was predicted as surprise (90%).
- Labels come from hashtags and are noisy.
- Single run per model, with no cross-validation or multiple seeds, so small differences between models are not statistically established.

## Future work

- Try softer class weights (e.g. square root of balanced weights) to improve precision on surprise and love
- Run multiple seeds and report mean ± standard deviation
- Compare with a pretrained transformer (e.g. fine-tuned DistilBERT)
- Add a Streamlit demo for interactive predictions

---

## Project structure

```
├── emotion_classification_v2.ipynb   # full pipeline: EDA, training, evaluation, export
├── Artifacts/
│   ├── emotion_model.keras           # trained best model
│   ├── tokenizer.pkl                 # fitted Keras tokenizer
│   └── config.json                   # max_len, label names, model name
├── images/
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook emotion_classification_v2.ipynb
```

`requirements.txt`:

```
tensorflow
datasets
scikit-learn
pandas
numpy
seaborn
matplotlib
```

## Using the saved model

```python
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("Artifacts/emotion_model.keras")
with open("Artifacts/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("Artifacts/config.json") as f:
    config = json.load(f)

def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=config["max_len"], padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(probs))
    return config["label_names"][idx], float(probs[idx])

print(predict_emotion("i feel so grateful for everything my family has done for me"))
# ('joy', 0.93)
```

## Tech stack

Python, TensorFlow/Keras, scikit-learn, Hugging Face Datasets, pandas, NumPy, Matplotlib, Seaborn

## Reference

Dataset: Saravia et al., *CARER: Contextualized Affect Representations for Emotion Recognition*, EMNLP 2018.
