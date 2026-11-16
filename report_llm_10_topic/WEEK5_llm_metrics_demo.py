"""Small, beginner-friendly LLM evaluation demonstration.

Prepared by DOEUN Bunheng — I5_AMS_A.
Run: python WEEK5_llm_metrics_demo.py

The core demonstrations use only Python's standard library.
Optional text-generation metrics require third-party packages.
"""

import re
import string


def exact_match(prediction: str, reference: str) -> int:
    """Return 1 when the two strings match exactly, otherwise 0."""
    return int(prediction == reference)


def normalize_english(text: str) -> str:
    """Normalize English short answers; not suitable for every language."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_score(predictions: list[str], references: list[str], normalized=False) -> float:
    """Average the per-question Exact Match scores."""
    if not references or len(predictions) != len(references):
        raise ValueError("Predictions and references must be equally sized and non-empty")
    if normalized:
        predictions = [normalize_english(x) for x in predictions]
        references = [normalize_english(x) for x in references]
    return sum(exact_match(p, r) for p, r in zip(predictions, references)) / len(references)


def classification_scores(truth: list[int], predictions: list[int]) -> dict:
    """Binary classification: Accuracy, Precision, Recall, F1 and confusion counts."""
    if not truth or len(truth) != len(predictions):
        raise ValueError("Ground truth and predictions must be equally sized and non-empty")
    if not set(truth + predictions).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")

    tp = sum(t == 1 and p == 1 for t, p in zip(truth, predictions))
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predictions))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": (tp + tn) / len(truth),
        "precision": precision, "recall": recall, "f1": f1,
    }


def edit_distance(reference_units, prediction_units) -> int:
    """Minimum insertions, deletions and substitutions (Levenshtein distance)."""
    previous = list(range(len(prediction_units) + 1))
    for i, ref in enumerate(reference_units, start=1):
        current = [i]
        for j, pred in enumerate(prediction_units, start=1):
            current.append(min(
                previous[j] + 1,               # Delete a reference unit
                current[j - 1] + 1,            # Insert a prediction unit
                previous[j - 1] + (ref != pred) # Keep or substitute
            ))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, prediction: str) -> float:
    """WER using whitespace-separated words for an instructional example."""
    ref_words, pred_words = reference.split(), prediction.split()
    if not ref_words:
        raise ValueError("Reference has no words")
    return edit_distance(ref_words, pred_words) / len(ref_words)


def character_error_rate(reference: str, prediction: str) -> float:
    """CER using Python Unicode code points as the character units."""
    if not reference:
        raise ValueError("Reference is empty")
    return edit_distance(reference, prediction) / len(reference)


def run_optional_metrics() -> None:
    """Demonstrate BLEU, ROUGE, METEOR if corresponding packages are available."""
    reference = "the cat sleeps"
    candidate = "the cat sleeps"

    try:
        import sacrebleu
        bleu = sacrebleu.sentence_bleu(candidate, [reference], use_effective_order=True)
        print(f"Sentence BLEU (0–100): {bleu.score:.2f}")
    except ImportError:
        print("Sentence BLEU skipped (install sacrebleu)")

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rouge = scorer.score(reference, candidate)
        print(f"ROUGE-L F1: {rouge['rougeL'].fmeasure:.4f}")
    except ImportError:
        print("ROUGE skipped (install rouge-score)")

    try:
        from nltk.translate.meteor_score import meteor_score
        score = meteor_score([reference.split()], candidate.split())
        print(f"METEOR: {score:.4f}")
    except (ImportError, LookupError) as error:
        print(f"METEOR skipped (install nltk / optional WordNet): {error.__class__.__name__}")


def run_bert_score() -> None:
    """Optional and slower: downloads a pretrained encoder if not cached."""
    try:
        from bert_score import score
    except ImportError:
        print("BERTScore skipped (install bert-score and torch)")
        return
    candidates = ["The kid feels joyful."]
    references = ["The boy is happy."]
    p, r, f = score(candidates, references, model_type="bert-base-uncased", num_layers=9)
    print(f"BERTScore F1: {f[0].item():.4f}")


def main() -> None:
    references = ["Paris", "4", "Jupiter", "H2O"]
    model_a = ["Paris", "4", "Jupiter", "H2O"]
    model_b = ["Paris", "Four", "Jupiter", "Water"]

    print("=== Exact Match: sample predictions ===")
    print(f"Model A: {em_score(model_a, references):.0%}")
    print(f"Model B: {em_score(model_b, references):.0%}")
    print(f"Paris/paris, strict: {em_score(['paris'], ['Paris']):.0%}")
    print(f"Paris/paris, normalized: {em_score(['paris'], ['Paris'], normalized=True):.0%}")

    print("\n=== Classification example ===")
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    for key, value in classification_scores(y_true, y_pred).items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")

    print("\n=== Word & character errors ===")
    print(f"WER: {word_error_rate('the cat sits on the mat', 'the cat sit on the mat'):.1%}")
    print(f"CER: {character_error_rate('kitten', 'sitting'):.1%}")

    print("\n=== Optional lexical metrics ===")
    run_optional_metrics()
    print("\nTo calculate BERTScore, call run_bert_score() separately; it may download model weights.")


if __name__ == "__main__":
    main()
