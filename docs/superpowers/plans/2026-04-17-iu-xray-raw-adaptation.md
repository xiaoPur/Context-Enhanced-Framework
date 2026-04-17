# IU X-Ray Raw Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make this repository runnable on AutoDL with raw Indiana IU X-Ray CSV data, automatic BLEU/ROUGE evaluation, and optional Qwen post-processing evaluation.

**Architecture:** Keep the existing multimodal model stack intact, add a raw-data dataset adapter plus utility modules around training and evaluation, and move all server-specific settings into CLI arguments. The label branch stays enabled by reusing the bundled legacy IU X-Ray label file and allowing optional external label overrides.

**Tech Stack:** Python 3.8, PyTorch 1.7, torchvision, SentencePiece, Pillow, pandas, pure-Python BLEU/ROUGE helpers, optional transformers for Qwen

---

### Task 1: Lock the target behavior with tests

**Files:**
- Create: `tests/test_indiana_raw_dataset.py`
- Create: `tests/test_evaluation.py`
- Create: `tests/test_qwen_postprocess.py`

- [ ] **Step 1: Write the failing dataset tests**

```python
def test_indiana_raw_dataset_builds_multiview_sample(tmp_path):
    dataset = IndianaRawIUXRAY(
        directory=str(tmp_path),
        reports_csv='indiana_reports.csv',
        projections_csv='indiana_projections.csv',
        images_dir='images_normalized',
    )
    sample_source, sample_target = dataset[0]
    assert sample_source[0][0].shape[0] == 2
    assert sample_source[3].shape[0] == dataset.max_len
    assert sample_target[0].shape[0] == dataset.max_len
    assert sample_target[1].shape[0] == 114
```

- [ ] **Step 2: Write the failing metric tests**

```python
def test_compute_report_metrics_returns_bleu_and_rouge():
    metrics = compute_report_metrics(
        references=['no acute cardiopulmonary abnormality .'],
        hypotheses=['no acute cardiopulmonary abnormality .'],
    )
    assert metrics['bleu_1'] == 1.0
    assert metrics['bleu_4'] == 1.0
    assert metrics['rouge_l'] == 1.0
```

- [ ] **Step 3: Write the failing Qwen prompt tests**

```python
def test_build_qwen_rewrite_prompt_contains_constraints():
    prompt = build_qwen_rewrite_prompt(
        history='cough . no prior study .',
        draft='lungs are clear .',
    )
    assert 'Only revise the report text' in prompt
    assert 'Do not introduce new findings' in prompt
    assert 'cough . no prior study .' in prompt
    assert 'lungs are clear .' in prompt
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run: `python -m unittest discover -s tests -v`

Expected: failures or import errors because the raw dataset adapter, metric helpers, and Qwen prompt builder do not exist yet.

### Task 2: Implement raw Indiana IU X-Ray dataset support

**Files:**
- Modify: `datasets.py`
- Create: `tests/test_indiana_raw_dataset.py`

- [ ] **Step 1: Add the raw dataset helper API**

```python
def normalize_report_text(text):
    text = '' if text is None else str(text)
    return ' '.join(text.replace('\n', ' ').strip().lower().split())


def legacy_uid_key(uid):
    return f'ecgen-radiology/{int(uid)}.xml'
```

- [ ] **Step 2: Add the new dataset class**

```python
class IndianaRawIUXRAY(data.Dataset):
    def __init__(self, directory, reports_csv='indiana_reports.csv',
                 projections_csv='indiana_projections.csv',
                 images_dir='images_normalized', input_size=(256, 256),
                 random_transform=True, max_views=2,
                 sources=('image', 'caption', 'label', 'history'),
                 targets=('caption', 'label'), max_len=1000,
                 vocab_file='iu_xray/nlmcxr_unigram_1000.model',
                 label_file='iu_xray/file2label.json', external_label_file=None,
                 split_file=None, split='train', train_size=0.7,
                 val_size=0.1, test_size=0.2, seed=123):
        ...
```

- [ ] **Step 3: Preserve the old sample shape contract**

```python
if self.sources[i] == 'image':
    sources.append((imgs, vpos))
if self.sources[i] == 'history':
    sources.append(source_info)
if self.sources[i] == 'label':
    sources.append(np.concatenate([base_labels, np_labels]))
if self.sources[i] == 'caption':
    sources.append(target_info)
```

- [ ] **Step 4: Support deterministic split persistence**

```python
def save_or_load_split(self):
    if self.split_file and os.path.exists(self.split_file):
        ...
    else:
        ...
```

- [ ] **Step 5: Run dataset-focused tests**

Run: `python -m unittest tests.test_indiana_raw_dataset -v`

Expected: PASS

### Task 3: Implement metric utilities and Qwen post-processing utilities

**Files:**
- Create: `evaluation.py`
- Create: `qwen_postprocess.py`
- Create: `tests/test_evaluation.py`
- Create: `tests/test_qwen_postprocess.py`

- [ ] **Step 1: Add pure-Python metric helpers**

```python
def compute_report_metrics(references, hypotheses):
    return {
        'bleu_1': corpus_bleu(references, hypotheses, (1.0, 0.0, 0.0, 0.0)),
        'bleu_2': corpus_bleu(references, hypotheses, (0.5, 0.5, 0.0, 0.0)),
        'bleu_3': corpus_bleu(references, hypotheses, (1/3, 1/3, 1/3, 0.0)),
        'bleu_4': corpus_bleu(references, hypotheses, (0.25, 0.25, 0.25, 0.25)),
        'rouge_l': mean_rouge_l(references, hypotheses),
    }
```

- [ ] **Step 2: Add result export helpers**

```python
def write_report_outputs(output_dir, references, hypotheses, metrics, records, prefix=''):
    ...
```

- [ ] **Step 3: Add Qwen prompt builder and lazy transformers loader**

```python
def build_qwen_rewrite_prompt(history, draft):
    return (
        'You are a radiology report editor...'
    )
```

- [ ] **Step 4: Run utility tests**

Run: `python -m unittest tests.test_evaluation tests.test_qwen_postprocess -v`

Expected: PASS

### Task 4: Refactor the training entrypoint around CLI arguments and evaluation

**Files:**
- Modify: `train.py`
- Modify: `models.py`
- Modify: `utils.py`

- [ ] **Step 1: Fix the generator inference path in `models.py`**

```python
def infer(self, source_embed, source_embed2, source_pad_mask=None, max_len=100,
          top_k=1, bos_id=1, pad_id=3):
    ...
```

- [ ] **Step 2: Replace global constants in `train.py` with argparse**

```python
parser.add_argument('--dataset-name', default='indiana_raw')
parser.add_argument('--data-root', required=True)
parser.add_argument('--output-dir', default='outputs')
parser.add_argument('--phase', choices=['train', 'infer'], default='train')
parser.add_argument('--run-eval', action='store_true')
parser.add_argument('--run-qwen-eval', action='store_true')
```

- [ ] **Step 3: Add dataset factory and evaluation integration**

```python
dataset = build_dataset_from_args(args)
...
references, hypotheses, records = generate_reports(...)
metrics = compute_report_metrics(references, hypotheses)
write_report_outputs(...)
```

- [ ] **Step 4: Add optional Qwen evaluation call**

```python
if args.run_qwen_eval:
    qwen_records = rewrite_reports_with_qwen(...)
    qwen_metrics = compute_report_metrics(...)
```

- [ ] **Step 5: Run a no-data smoke check on imports and CLI**

Run: `python train.py --help`

Expected: exit code 0 and argument help text printed.

### Task 5: Document the AutoDL workflow and verify the repository locally

**Files:**
- Create: `docs/iu_xray_autodl_runbook_zh.md`
- Modify: `README.md`

- [ ] **Step 1: Write the Chinese AutoDL runbook**

```markdown
## 环境准备
## 数据目录
## 训练命令
## 原始评估命令
## Qwen 后处理评估命令
## 常见问题
```

- [ ] **Step 2: Add a short top-level README pointer**

```markdown
For raw IU X-Ray adaptation and AutoDL usage, see `docs/iu_xray_autodl_runbook_zh.md`.
```

- [ ] **Step 3: Run the full local verification suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 4: Run a syntax/import verification pass**

Run: `@' ... compile(...) ... '@ | python -`

Expected: all edited Python files compile without syntax errors.
