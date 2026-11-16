# LLM Performance Evaluation — Topics 1 to 10

**Prepared by:** DOEUN Bunheng  
**Class:** I5_AMS_A  
**School:** Institute of Technology of Cambodia (ITC)  
**Course:** Advanced Programming / LLM Performance Evaluation  
**Date:** 16 November 2026

> My notes from the ten group presentations about evaluating Large Language Models (LLMs). I wrote this README to keep the main ideas, formulas, examples, and useful Python functions in one place. It is a learning summary, **not** a new experiment comparing ten models.

## Contents

- [Why do we evaluate LLMs?](#why-do-we-evaluate-llms)
- [Quick comparison](#quick-comparison)
- [1. Exact Match (EM)](#1-exact-match-em)
- [2. Accuracy](#2-accuracy)
- [3. Precision, Recall, and F1-score](#3-precision-recall-and-f1-score)
- [4. BLEU](#4-bleu)
- [5. ROUGE](#5-rouge)
- [6. METEOR](#6-meteor)
- [7. BERTScore](#7-bertscore)
- [8. Word Error Rate (WER)](#8-word-error-rate-wer)
- [9. Character Error Rate (CER)](#9-character-error-rate-cer)
- [10. Which metric should we choose?](#10-which-metric-should-we-choose)
- [What the group experiments showed](#what-the-group-experiments-showed)
- [Python setup and mini demo](#python-setup-and-mini-demo)
- [My final takeaways](#my-final-takeaways)
- [Sources](#sources)

---

## Why do we evaluate LLMs?

An LLM can answer a question, classify a review, translate a sentence, or write a summary. But the meaning of **“good output”** depends on the task.

For example:

- If we ask for a country code, we probably want **the exact answer**.
- If we classify a review as positive or negative, we want **the correct class**.
- If we translate or summarize, several sentences can be **correct even with different wording**.
- If we transcribe speech or recognize text from an image, we want to know **how many words or characters are wrong**.

Most reference-based evaluations start with three things:

| Term | Meaning | Example |
|---|---|---|
| **Input** | The question or task given to the model | “What is the capital of France?” |
| **Reference** | The expected answer or correct label | `Paris` |
| **Prediction / Candidate** | The answer produced by the model | `The capital is Paris.` |

The basic process is:

```text
Input question
     ↓
    LLM
     ↓
Model prediction ──┐
                   ├──→ Evaluation metric ──→ Score
Reference answer ──┘
```

**Important:** A similarity score is not always a correctness score. A model can copy the reference wording and still misunderstand the task, while another model can give the right meaning in different words.

## Quick comparison

There are **ten group topics**, but Group 3 covers three connected classification metrics and Group 10 compares the others.

| Group | Topic | What it measures | Better score | Common application |
|---|---|---|---|---|
| 1 | Exact Match | Identical answer strings | Higher | Short-answer QA |
| 2 | Accuracy | Correct predictions / labels | Higher | Classification |
| 3 | Precision / Recall / F1 | False positives and false negatives | Higher | Sentiment, spam, screening |
| 4 | BLEU | Candidate n-gram overlap | Higher | Machine translation |
| 5 | ROUGE | Reference word/sequence coverage | Higher | Summarization |
| 6 | METEOR | Words, stems, synonyms, and order | Higher | Translation |
| 7 | BERTScore | Contextual token similarity | Usually higher* | Paraphrasing and summaries |
| 8 | WER | Word insertions, deletions, substitutions | **Lower** | Speech-to-text |
| 9 | CER | Character insertions, deletions, substitutions | **Lower** | OCR and transcription |
| 10 | Comparative evaluation | Choosing a metric for the task | — | Evaluation design |

*BERTScore values depend on the encoder and settings. Do not compare raw numbers from different metric types as though they have the same meaning.*

---

## 1. Exact Match (EM)

**The question EM asks:** *Is the predicted answer exactly the same as the reference?*

Each answer receives a binary score: **1** for a match and **0** otherwise. The final EM score is the average across all questions.

$$
EM = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(P_i = R_i)
$$

- $N$: number of questions
- $P_i$: model prediction
- $R_i$: reference answer
- $\mathbf{1}$: 1 when the two answers match; otherwise 0

### Simple example

| Reference | Prediction | Strict EM |
|---|---|---:|
| `Paris` | `Paris` | 1 |
| `Paris` | `paris` | 0 |
| `Phnom Penh` | `Phnom Penh!` | 0 |
| `4` | `Four` | 0 |

Even `4` and `Four` get a **strict EM of 0**, although their meaning is the same.

**Normalized EM** is a variation where we clean *both* answers first. A common English normalization removes capitalization differences, punctuation, articles (`a`, `an`, `the`), and extra spaces. The exact rules must be reported. Normalization helps with formatting; it does **not** understand paraphrases.

### Important Python functions

```python
import re
import string


def exact_match(prediction: str, reference: str) -> int:
    """Return 1 for identical strings and 0 otherwise."""
    return int(prediction == reference)


def normalize_english(text: str) -> str:
    """Example English-only normalization for short answers."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_score(predictions, references, normalized=False):
    if not references or len(predictions) != len(references):
        raise ValueError("Predictions and references must be equally sized and non-empty")
    if normalized:
        predictions = [normalize_english(p) for p in predictions]
        references = [normalize_english(r) for r in references]
    return sum(exact_match(p, r) for p, r in zip(predictions, references)) / len(references)


print(exact_match("Paris", "Paris"))            # 1
print(exact_match("Paris", "paris"))            # 0
print(em_score(["paris"], ["Paris"], True))     # 1.0
```

**Use it for:** extractive question answering, codes, IDs, and tightly formatted answers.  
**Be careful:** it gives no partial credit and can reject a correct rephrasing.

---

## 2. Accuracy

Accuracy answers: *Out of all predictions, how many were correct?*

It is most useful when the LLM predicts a **class label**, such as `Positive` or `Negative`.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

For the four outcomes:

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | TP — correct positive | FN — missed positive |
| **Actually negative** | FP — false alarm | TN — correct negative |

If 90 of 100 classifications are right, accuracy is **90%**.

```python
from sklearn.metrics import accuracy_score

y_true = ["positive", "negative", "positive"]
y_pred = ["positive", "positive", "positive"]

print(accuracy_score(y_true, y_pred))  # 0.666666...
```

**Use it for:** sentiment classification and multiple-choice questions with known labels.  
**Be careful:** 95% accuracy can still be misleading if 95% of the dataset belongs to one class. Check the class balance and the confusion matrix.

---

## 3. Precision, Recall, and F1-score

I find these three easier to remember as **three different questions**:

- **Precision:** When the model predicts *positive*, how often is it right?
- **Recall:** Of all truly positive cases, how many did the model find?
- **F1:** How well does the model balance precision and recall?

$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

$$
F1 = 2\frac{Precision \times Recall}{Precision + Recall}
$$

### Worked example: spam detection

Suppose $TP=20$, $FP=5$, $FN=10$, and $TN=65$.

- Precision = $20/(20+5)$ = **80.0%**
- Recall = $20/(20+10)$ = **66.7%**
- F1 = $2(0.8)(0.6667)/(0.8+0.6667)$ ≈ **72.7%**

The model is reasonably precise when it flags spam, but it still **misses some actual spam**.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]

print("Precision:", precision_score(y_true, y_pred))  # 0.75
print("Recall:", recall_score(y_true, y_pred))        # 0.75
print("F1:", f1_score(y_true, y_pred))                # 0.75
```

**Use them for:** spam detection, sentiment analysis, and other classification problems where false alarms and missed cases have different costs.  
**Be careful:** for multiclass classification, specify whether you use **macro**, **micro**, or **weighted** averaging. F1 does not directly count true negatives.

---

## 4. BLEU

**BLEU** stands for *Bilingual Evaluation Understudy*. It is mainly associated with **machine translation**.

BLEU checks whether consecutive word sequences (**n-grams**) in the model output also appear in the reference. It also applies a **brevity penalty** so an extremely short answer does not score too well.

| N-gram | Example from `the cat is sleeping` |
|---|---|
| 1-gram | `the`, `cat`, `is`, `sleeping` |
| 2-gram | `the cat`, `cat is`, `is sleeping` |
| 3-gram | `the cat is`, `cat is sleeping` |

$$
BLEU = BP\times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

Here, $p_n$ is **clipped n-gram precision**, $w_n$ is an n-gram weight, and $BP$ is the brevity penalty.

### Example from Group 4 (simplified BLEU-2)

- Reference: `the cat is on the mat`
- Candidate: `the cat is on mat`
- Unigram precision = $5/5 = 1$
- Bigram precision = $3/4 = 0.75$
- Brevity penalty ≈ $0.819$

Therefore, **BLEU-2 ≈ 0.709 (70.9/100)**. This is a *teaching calculation*, not standard corpus BLEU-4.

```python
import sacrebleu

reference = "the cat is on the mat"
candidate = "the cat is on mat"

score = sacrebleu.sentence_bleu(
    candidate, [reference], use_effective_order=True
)
print("Sentence BLEU:", score.score)  # Score on a 0–100 scale
```

**Use it for:** comparing machine-translation systems on the **same test set**.  
**Be careful:** a correct translation can use synonyms and still receive a lower BLEU score. For formal comparisons, prefer corpus-level BLEU and document tokenization and smoothing.

---

## 5. ROUGE

**ROUGE** stands for *Recall-Oriented Understudy for Gisting Evaluation*. It is often used to evaluate **text summaries**.

Where BLEU focuses on candidate n-gram precision, ROUGE often asks how much of the **reference content** appears in the generated text.

Main variants covered in the slides:

| Variant | Main idea |
|---|---|
| **ROUGE-1** | Single-word overlap |
| **ROUGE-2** | Two-word sequence overlap |
| **ROUGE-L** | Longest common subsequence (LCS) |
| **ROUGE-Lsum** | Sentence-aware summary-level LCS |
| **ROUGE-W** | Weighted LCS |
| **ROUGE-S / SU** | Skip-bigrams, optionally with unigrams |

For ROUGE-N recall:

$$
ROUGE\text{-}N_{recall} = \frac{\text{matched reference n-grams}}{\text{total reference n-grams}}
$$

### Simple example

- Reference: `the weather is cold outside`
- Candidate: `the weather is cold`

Four of the five reference words are recovered. So **ROUGE-1 recall = 4/5 = 0.80**, while unigram precision is **4/4 = 1.00** (using simple whitespace tokenization).

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)

scores = scorer.score(
    "the weather is cold outside",
    "the weather is cold"
)

print("ROUGE-1 recall:", scores["rouge1"].recall)
print("ROUGE-L F1:", scores["rougeL"].fmeasure)
```

**Use it for:** summarization and checking reference coverage.  
**Be careful:** high word overlap does not guarantee a factual or readable summary.

---

## 6. METEOR

**METEOR** stands for *Metric for Evaluation of Translation with Explicit ORdering*.

It tries to improve on simple word overlap by matching:

1. Exact words
2. Related word forms or stems
3. Synonyms, where language resources allow them

METEOR also considers **word order**. Matching words split into many separated groups receive a penalty.

$$
P = \frac{m}{\text{candidate words}}, \qquad R = \frac{m}{\text{reference words}}
$$

$$
F = \frac{10PR}{R + 9P}, \qquad Penalty = 0.5\left(\frac{chunks}{m}\right)^3
$$

$$
METEOR = F(1 - Penalty)
$$

Here $m$ is the number of aligned words, and `chunks` counts the contiguous matching groups. The displayed formula is the one presented by Group 6.

**Group 6 examples:**

- `the cat sleeps` vs. `the cat sleeps` → METEOR ≈ **0.9815**
- `the cat sleeps` vs. `sleeps cat the` → METEOR = **0.5000**

Both candidates contain the same words, but the second has a different word order.

```python
from nltk.translate.meteor_score import meteor_score

reference = "the cat sleeps".split()
candidate = "the cat sleeps".split()

print(round(meteor_score([reference], candidate), 4))  # 0.9815
```

For synonym matching, the NLTK implementation may require the WordNet dataset:

```python
import nltk
nltk.download("wordnet")
```

**Use it for:** translation and other reference-based tasks that benefit from stems, synonyms, and ordering.  
**Be careful:** its synonym behavior depends on language resources; English-oriented resources may not work well for Khmer.

---

## 7. BERTScore

BERTScore compares **contextual meaning**, not just whether the same words appear.

It passes the reference and candidate through a pretrained transformer (such as BERT), represents each token as a vector, and compares token vectors using **cosine similarity**. Each token keeps its best match from the other sentence.

For example:

- Reference: `The boy is happy.`
- Candidate: `The kid feels joyful.`

The words are different, but `boy` / `kid` and `happy` / `joyful` are related. BERTScore can give credit for this, unlike strict EM.

$$
P_{BERT} = \operatorname{avg}_{c\in C}\max_{r\in R} sim(c,r)
$$

$$
R_{BERT} = \operatorname{avg}_{r\in R}\max_{c\in C} sim(r,c)
$$

$$
F1_{BERT} = \frac{2P_{BERT}R_{BERT}}{P_{BERT}+R_{BERT}}
$$

The Group 7 presentation also discusses optional **IDF weighting** and **baseline rescaling**. Results depend on the chosen encoder and layer.

```python
from bert_score import score

candidates = ["The kid feels joyful."]
references = ["The boy is happy."]

precision, recall, f1 = score(
    candidates,
    references,
    model_type="bert-base-uncased",
    num_layers=9
)

print("BERTScore F1:", f1.tolist())
```

> **Note:** BERTScore is more computationally expensive. This example needs `bert-score`, PyTorch, and a pretrained encoder download on first use.

**Use it for:** paraphrasing, abstractive summarization, and open-ended answers with valid alternative wording.  
**Be careful:** semantically similar text can still contain a factual error or negation. BERTScore is **not a fact checker**.

---

## 8. Word Error Rate (WER)

WER is commonly used for **speech-to-text systems**. It measures how many word-level edits are needed to change a reference transcription into a predicted transcription.

There are three edit types:

- **S — Substitution:** wrong word
- **D — Deletion:** missing word
- **I — Insertion:** extra word

$$
WER = \frac{S + D + I}{N_{words}}
$$

where $N_{words}$ is the number of words in the reference. **Lower is better**, and WER can exceed 100% if there are many insertions.

### Example

- Reference: `I love to read books`
- Prediction: `I love to read book`

One word is substituted among five reference words, so **WER = 1/5 = 20%**.

```python
from jiwer import wer

reference = "the cat sits on the mat"
prediction = "the cat sit on the mat"

print(f"WER: {wer(reference, prediction):.1%}")  # 16.7%
```

**Use it for:** speech recognition, subtitles, and transcripts.  
**Be careful:** it checks words, not meaning. Word segmentation and normalization strongly influence results; **do not assume** `1 - WER` equals normal classification accuracy.

---

## 9. Character Error Rate (CER)

CER uses the **same edit-distance idea as WER**, but compares characters rather than words. This makes it useful for **OCR, transcription, spelling correction, and precise text extraction**.

$$
CER = \frac{S + D + I}{N_{characters}}
$$

| Reference | Prediction | Character error | CER |
|---|---|---|---:|
| `cat` | `cut` | 1 substitution | 33.33% |
| `cat` | `cart` | 1 insertion | 33.33% |
| `kitten` | `sitting` | 3 edits | 50.00% |

```python
from jiwer import cer

reference = "kitten"
prediction = "sitting"

print(f"CER: {cer(reference, prediction):.2%}")  # 50.00%
```

### WER vs. CER

If the reference is `hello` and the output is `helo`:

- **WER = 100%** because the only word is different.
- **CER = 20%** because just one of five reference characters is missing.

**Use it for:** OCR and tasks where a missing character matters.  
**Be careful:** case, spaces, punctuation, Unicode normalization, and the definition of a character can affect CER, particularly for scripts such as Khmer.

---

## 10. Which metric should we choose?

Group 10 makes the most important overall point: **there is no single best LLM metric for every task**.

| Task | Good starting combination | Why |
|---|---|---|
| Short-answer question answering | Normalized EM + token F1 | Checks exact answers and partial word overlap |
| Sentiment or label classification | Accuracy + Precision/Recall/F1 | Checks total correctness and class-specific errors |
| Machine translation | Corpus BLEU + METEOR or BERTScore | Combines wording and broader similarity |
| Text summarization | ROUGE + BERTScore + factual checks | Checks coverage, related meaning, and facts |
| Speech recognition | WER + CER + listening | Shows word and character transcription errors |
| Optical character recognition | CER + WER + visual review | Detects transcription mistakes |
| Creative / open-ended generation | Human rubric + appropriate semantic metrics | Exact wording alone cannot judge quality |

### Three mistakes to avoid

1. **Mixing score meanings:** BLEU = 70 is not the same as 70% accuracy or 0.70 BERTScore.
2. **Treating similarity as truth:** overlapping words do not prove that facts are correct.
3. **Comparing unfair experiments:** two models must be evaluated with the same test data, scoring setup, and documented preprocessing.

---

## What the group experiments showed

These are figures **reported in the uploaded group presentations**. They are not a combined leaderboard, and I did not independently rerun the experiments.

| Group | Example or experiment in the slides | How to interpret it |
|---|---|---|
| 1 — EM | Sample Model A **4/4 (100%)**, Model B **2/4 (50%)** | **Illustrative** manually specified answers |
| 2 — Accuracy | Gemini 1.5 and Llama 3 each **4/5 (80%)** on five reviews | **Source-reported** classification comparison |
| 3 — P/R/F1 | Spam and screening confusion matrices | **Illustrative** worked calculations |
| 4 — BLEU | Simplified BLEU-2 ≈ **70.9/100** | **Teaching example**, not corpus BLEU-4 |
| 5 — ROUGE | Word overlap and LCS examples | **Conceptual** demonstrations |
| 6 — METEOR | Sample answers ≈ **0.999** vs. **0.536** | **Illustrative** rather than verified live-model outputs |
| 7 — BERTScore | Qwen and SmolLM2 paraphrasing BERTScore F1: **0.9133** vs. **1.0000** | **Source-reported local** comparison; copying a reference can score perfectly |
| 8 — WER | Whisper-base **9.4%** vs. Whisper-tiny **18.8%** | **Source-reported**, five sentences and 32 reference words |
| 9 — CER | OCR table: Tesseract **20.55%**, Netra-OCR **5.87%**, Chakku **4.50%** | **Source-reported** CER figures, not independently checked |
| 10 — Comparison | Cross-metric recommendations | Theory synthesis; **not** a new benchmark |

The main lesson is that a strong score is meaningful only **for a defined task, reference set, and evaluation procedure**.

---

## Python setup and mini demo

### 1. Install the common libraries

Run in a terminal or a Jupyter Notebook cell (use `%pip` in Jupyter):

```bash
python -m pip install scikit-learn sacrebleu rouge-score nltk jiwer
```

For BERTScore, install the additional package (PyTorch may also be installed as a dependency):

```bash
python -m pip install bert-score torch
```

### 2. Run the companion demonstration

I included a small **[`WEEK5_llm_metrics_demo.py`](report_llm_10_topic/WEEK5_llm_metrics_demo.py)** file with runnable functions and simple tests. Its basic string/classification/edit-distance examples use only Python's standard library; it also includes optional demonstrations for BLEU, ROUGE, METEOR, and BERTScore.

```bash
python report_llm_10_topic/WEEK5_llm_metrics_demo.py
```

In Jupyter Notebook, you can copy the code examples from this README into individual cells instead.

### 3. Example: compare two answer lists using EM

```python
references = ["Paris", "4", "Jupiter", "H2O"]
model_a = ["Paris", "4", "Jupiter", "H2O"]
model_b = ["Paris", "Four", "Jupiter", "Water"]

print("Model A:", em_score(model_a, references))  # 1.0 = 100%
print("Model B:", em_score(model_b, references))  # 0.5 = 50%
```

These are **example strings**, not outputs from actual LLM API calls. To do a real experiment, save predictions from two named model versions for the *same* set of questions, then calculate the metrics.

### A checklist before reporting any results

- [ ] Use the same questions and references for each model.
- [ ] Save the original model outputs (do not edit them after generation).
- [ ] Record model versions, prompt wording, generation settings, and run date.
- [ ] Document tokenization, normalization, and whether multiple references are accepted.
- [ ] Report the dataset size, class balance, and relevant languages.
- [ ] Inspect errors, including valid paraphrases and factual contradictions.
- [ ] Keep **illustrative examples** separate from **measured experimental results**.

### A note about Khmer evaluation

Some English preprocessing rules do not transfer directly to Khmer. In particular, English article removal, whitespace-based word splitting, and WordNet synonym matching should not be applied without checking whether they are appropriate. For Khmer word-level BLEU, ROUGE, or WER, word segmentation matters; for CER, Unicode and character-counting choices should be documented.

---

## My final takeaways

After reviewing all ten topics, this is how I remember them:

- **EM** checks whether the whole answer is exactly the same.
- **Accuracy** tells me how often a classifier is correct.
- **Precision** looks at false alarms; **Recall** looks at missed positives; **F1** balances them.
- **BLEU** looks at candidate n-gram matches, especially for translation.
- **ROUGE** measures reference coverage, especially for summaries.
- **METEOR** also considers stems, synonyms, and word order.
- **BERTScore** compares contextual token meaning, but not factual truth.
- **WER and CER** count editing mistakes at word and character level.
- **Group 10** shows why we should choose and combine metrics based on the task.

**The one point I want to remember:** *A metric is useful only when I understand exactly what it measures — and what it cannot measure.*

## Sources

This README is adapted from my combined report, **“Evaluation Methods for Large Language Models (LLMs): A Comparative Review of Group Topics 1–10”**, which was prepared from the ten classroom presentations:

1. Group 1 — *Exact Match (EM) Evaluation for Large Language Models*
2. Group 2 — *Accuracy Evaluation*
3. Group 3 — *Precision, Recall, and F1-score*
4. Group 4 — *BLEU Score*
5. Group 5 — *ROUGE Score*
6. Group 6 — *METEOR Score*
7. Group 7 — *BERTScore: Evaluating Semantic Similarity in LLMs*
8. Group 8 — *Word Error Rate (WER)*
9. Group 9 — *Character Error Rate (CER)*
10. Group 10 — *Comparative Evaluation of LLM Metrics*

Original research and documentation mentioned in those presentations:

- Papineni et al. (2002), [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- Lin (2004), [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
- Banerjee & Lavie (2005), [METEOR: An Automatic Metric for MT Evaluation](https://aclanthology.org/W05-0909/)
- Zhang et al. (2020), [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
- Rajpurkar et al. (2016), [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
- [JiWER documentation](https://jitsi.github.io/jiwer/)

---

*Prepared for learning and classroom discussion by DOEUN Bunheng — I5_AMS_A.*
