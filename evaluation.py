import json
import math
import os
from collections import Counter


def _tokenize(text):
    return str(text).strip().split()


def _extract_ngrams(tokens, order):
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _closest_reference_length(reference_tokens, hypothesis_length):
    ref_length = len(reference_tokens)
    return ref_length


def corpus_bleu(references, hypotheses, weights):
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    max_order = len(weights)
    clipped_counts = [0 for _ in range(max_order)]
    total_counts = [0 for _ in range(max_order)]
    reference_length = 0
    hypothesis_length = 0

    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = _tokenize(reference)
        hyp_tokens = _tokenize(hypothesis)
        reference_length += _closest_reference_length(ref_tokens, len(hyp_tokens))
        hypothesis_length += len(hyp_tokens)

        for order in range(1, max_order + 1):
            ref_ngrams = _extract_ngrams(ref_tokens, order)
            hyp_ngrams = _extract_ngrams(hyp_tokens, order)
            clipped_counts[order - 1] += sum((hyp_ngrams & ref_ngrams).values())
            total_counts[order - 1] += max(len(hyp_tokens) - order + 1, 0)

    if hypothesis_length == 0:
        return 0.0

    precisions = []
    for order, weight in enumerate(weights, start=1):
        if weight == 0.0:
            precisions.append(1.0)
            continue
        if total_counts[order - 1] == 0 or clipped_counts[order - 1] == 0:
            return 0.0
        precisions.append(clipped_counts[order - 1] / total_counts[order - 1])

    brevity_penalty = 1.0
    if hypothesis_length < reference_length:
        brevity_penalty = math.exp(1.0 - (reference_length / hypothesis_length))

    score = 0.0
    for precision, weight in zip(precisions, weights):
        if weight:
            score += weight * math.log(precision)
    return round(brevity_penalty * math.exp(score), 6)


def _lcs_length(reference_tokens, hypothesis_tokens):
    rows = len(reference_tokens) + 1
    cols = len(hypothesis_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]

    for row in range(1, rows):
        for col in range(1, cols):
            if reference_tokens[row - 1] == hypothesis_tokens[col - 1]:
                dp[row][col] = dp[row - 1][col - 1] + 1
            else:
                dp[row][col] = max(dp[row - 1][col], dp[row][col - 1])
    return dp[-1][-1]


def rouge_l(reference, hypothesis):
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 6)


def compute_report_metrics(references, hypotheses):
    return {
        "bleu_1": corpus_bleu(references, hypotheses, (1.0, 0.0, 0.0, 0.0)),
        "bleu_2": corpus_bleu(references, hypotheses, (0.5, 0.5, 0.0, 0.0)),
        "bleu_3": corpus_bleu(references, hypotheses, (1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0)),
        "bleu_4": corpus_bleu(references, hypotheses, (0.25, 0.25, 0.25, 0.25)),
        "rouge_l": round(sum(rouge_l(ref, hyp) for ref, hyp in zip(references, hypotheses)) / len(references), 6)
        if references
        else 0.0,
    }


def write_report_outputs(output_dir, references, hypotheses, metrics, records, prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = f"{prefix}_" if prefix else ""
    with open(os.path.join(output_dir, f"{file_prefix}references.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(references) + ("\n" if references else ""))
    with open(os.path.join(output_dir, f"{file_prefix}hypotheses.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(hypotheses) + ("\n" if hypotheses else ""))
    with open(os.path.join(output_dir, f"{file_prefix}metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, f"{file_prefix}predictions.jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
