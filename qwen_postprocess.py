def build_qwen_rewrite_prompt(history, draft):
    history = history or ""
    draft = draft or ""
    return (
        "You are a radiology report editor.\n"
        "Only revise the report text.\n"
        "Keep the report clinically faithful to the draft.\n"
        "Do not introduce new findings.\n"
        "Do not remove findings that are already present unless they are redundant wording.\n"
        "Keep the output concise and in plain report style.\n\n"
        f"Clinical context:\n{history}\n\n"
        f"Draft findings:\n{draft}\n\n"
        "Return only the revised findings text."
    )


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
        updated_record = dict(record)
        updated_record["qwen_hypothesis"] = revised_text or record.get("hypothesis", "")
        updated_record["qwen_prompt"] = prompt
        rewritten_records.append(updated_record)
    return rewritten_records
