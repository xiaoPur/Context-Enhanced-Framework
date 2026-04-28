import re


_METRIC_PUNCTUATION = re.compile(r"([.,;:!?()])")


def build_qwen_rewrite_prompt(history, draft):
    history = history or ""
    draft = draft or ""
    return (
        "You are a conservative radiology report copy editor.\n"
        "Your goal is to make the smallest possible edits to the draft findings.\n"
        "Preserve the draft wording, sentence order, medical terms, and report structure as much as possible.\n\n"
        "Rules:\n"
        "1. If the draft is understandable, return it unchanged.\n"
        "2. Do not paraphrase normal findings or replace terms with synonyms.\n"
        "3. Do not summarize, expand, merge, or reorder sentences.\n"
        "4. Do not introduce new findings.\n"
        "5. Do not remove findings from the draft, including normal and abnormal findings.\n"
        "6. Use the clinical context only to resolve ambiguity, not to add content.\n"
        "7. Preserve all xxxx placeholders exactly. Do not guess redacted words.\n"
        "8. Only fix obvious spelling, grammar, punctuation, spacing, or exact duplicate consecutive statements.\n"
        "9. Return only the final findings text.\n\n"
        f"Clinical context:\n{history}\n\n"
        f"Draft findings:\n{draft}\n\n"
        "Final findings:"
    )


def normalize_report_text_for_metrics(text):
    normalized = str(text or "").strip().lower()
    normalized = _METRIC_PUNCTUATION.sub(r" \1 ", normalized)
    return " ".join(normalized.split())


def build_qwen_output_record(record, revised_text):
    final_text = (revised_text or "").strip() or record.get("hypothesis", "")
    updated_record = dict(record)
    updated_record["qwen_hypothesis"] = final_text
    updated_record["qwen_hypothesis_normalized"] = normalize_report_text_for_metrics(final_text)
    return updated_record


def _load_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for Qwen post-processing. Install it in the optional Qwen environment."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def rewrite_reports_with_qwen(
    records,
    model_name_or_path,
    max_new_tokens=256,
    temperature=0.0,
    device_map="auto",
):
    AutoModelForCausalLM, AutoTokenizer = _load_transformers()
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )

    rewritten_records = []
    for record in records:
        prompt = build_qwen_rewrite_prompt(record.get("history", ""), record.get("hypothesis", ""))
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )
        new_tokens = generated[:, input_ids.shape[-1] :]
        revised_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        updated_record = build_qwen_output_record(record, revised_text)
        updated_record["qwen_prompt"] = prompt
        rewritten_records.append(updated_record)
    return rewritten_records
