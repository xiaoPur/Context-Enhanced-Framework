# IU X-Ray AutoDL 运行手册

这份手册只说明服务器上怎么跑，以及每一步命令对应到代码里做了什么。默认入口是 `run_indiana_raw.py`，不要再用旧的 `train.py`。

## 1. 目录准备

建议服务器目录保持一致：

```text
/root/autodl-tmp/Context-Enhanced-Framework/
/root/autodl-tmp/IU x-ray/
/root/autodl-tmp/IU x-ray/images_normalized/
/root/autodl-tmp/IU x-ray/indiana_reports.csv
/root/autodl-tmp/IU x-ray/indiana_projections.csv
```

训练和评估输出建议放到：

```text
/root/autodl-tmp/Context-Enhanced-Framework/artifacts/iu_xray_raw/
```

对应代码：

- `run_indiana_raw.py::parse_args()` 读取 `--data-root`、`--output-dir`、CSV 文件名和图像目录参数。
- `datasets.py::IndianaRawIUXRAY` 从 `indiana_reports.csv` 读取 `findings`，从 `indiana_projections.csv` 找图像文件。
- `run_indiana_raw.py::build_dataset_triplet()` 创建 train / val / test 三个数据集。

## 2. 安装主模型依赖

进入项目目录：

```bash
cd /root/autodl-tmp/Context-Enhanced-Framework
```

检查基础环境：

```bash
python --version
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

安装主模型依赖：

```bash
pip install pandas pillow sentencepiece scikit-learn tqdm
```

对应代码：

- `datasets.py` 依赖 `pandas` 读取 Indiana CSV。
- `datasets.py` 依赖 `Pillow` 读取 X-Ray 图像。
- `datasets.py` 依赖 `sentencepiece` 编码和解码报告文本。
- `run_indiana_raw.py::build_model()` 使用 `torch`、`torchvision.models.densenet121` 构建主模型。

如果 `torchvision` 已经随镜像安装，不要随便升级 PyTorch / torchvision，避免破坏服务器镜像环境。

## 3. 可选：安装 METEOR / CIDEr 评估依赖

只有当命令里使用 `--include-paper-metrics` 时，才需要这一节。

```bash
java -version
pip install pycocoevalcap nltk
python -c "from pycocoevalcap.meteor.meteor import Meteor; from pycocoevalcap.cider.cider import Cider; print('paper metrics ok')"
```

如果 `java -version` 不可用，并且服务器有 apt 权限：

```bash
apt-get update && apt-get install -y default-jre
```

对应代码：

- `run_indiana_raw.py::parse_args()` 读取 `--include-paper-metrics`。
- `evaluation.py::compute_report_metrics()` 在 `include_paper_metrics=True` 时追加论文指标。
- `evaluation.py::compute_paper_metrics()` 调用 `pycocoevalcap` 的 `Meteor` 和 `Cider`。
- 如果依赖没装却打开 `--include-paper-metrics`，代码会直接报错提醒安装服务器端评估依赖。

## 4. 训练并自动评估

推荐训练命令：

```bash
cd /root/autodl-tmp/Context-Enhanced-Framework

python run_indiana_raw.py \
  --phase train \
  --dataset-name indiana_raw \
  --data-root "/root/autodl-tmp/IU x-ray" \
  --output-dir "/root/autodl-tmp/Context-Enhanced-Framework/artifacts/iu_xray_raw" \
  --reports-csv indiana_reports.csv \
  --projections-csv indiana_projections.csv \
  --images-dir images_normalized \
  --batch-size 8 \
  --eval-batch-size 32 \
  --epochs 50 \
  --num-workers 8 \
  --pretrained-backbone \
  --run-eval
```

如果已经安装好 METEOR / CIDEr 依赖，并希望训练结束后一起输出论文指标，在末尾加：

```bash
  --include-paper-metrics
```

这一步对应代码：

- `main()` 调用 `seed_everything()` 固定随机种子。
- `build_dataset_triplet()` 创建 train / val / test 数据集，并把划分写入 `indiana_raw_splits.json`。
- `build_dataloaders()` 创建 PyTorch dataloader。
- `build_model()` 创建 DenseNet121 + MVCNN + TNN + Classifier + Generator + Context。
- `utils.py::train()` 执行训练。
- `utils.py::test()` 在 val / test 上计算 loss。
- `utils.py::save()` 保存验证集 loss 最低的 checkpoint。
- 因为命令带了 `--run-eval`，训练结束后会进入 `run_evaluation()`。
- `run_evaluation()` 调用 `evaluate_generation()` 生成报告，再调用 `write_report_outputs()` 写出结果。

训练输出重点看：

```text
artifacts/iu_xray_raw/checkpoints/
artifacts/iu_xray_raw/indiana_raw_splits.json
artifacts/iu_xray_raw/metrics.json
artifacts/iu_xray_raw/predictions.jsonl
artifacts/iu_xray_raw/references.txt
artifacts/iu_xray_raw/hypotheses.txt
```

`metrics.json` 默认包含：

```json
{
  "bleu_1": 0.0,
  "bleu_2": 0.0,
  "bleu_3": 0.0,
  "bleu_4": 0.0,
  "rouge_l": 0.0
}
```

如果加了 `--include-paper-metrics`，还会包含：

```json
{
  "meteor": 0.0,
  "cider": 0.0
}
```

## 5. 只用已有 checkpoint 重新评估

如果模型已经训练好，只想重新生成报告和指标：

```bash
cd /root/autodl-tmp/Context-Enhanced-Framework

python run_indiana_raw.py \
  --phase infer \
  --dataset-name indiana_raw \
  --data-root "/root/autodl-tmp/IU x-ray" \
  --output-dir "/root/autodl-tmp/Context-Enhanced-Framework/artifacts/iu_xray_raw" \
  --checkpoint-path "/root/autodl-tmp/Context-Enhanced-Framework/artifacts/iu_xray_raw/checkpoints/indiana_raw_Context_DenseNet121_MaxView2_NumLabel114_History.pt" \
  --num-workers 8 \
  --run-eval
```

如果要算 METEOR / CIDEr：

```bash
  --include-paper-metrics
```

这一步对应代码：

- `main()` 在 `--phase infer` 下通过 `utils.py::load()` 加载 checkpoint。
- `run_evaluation()` 进入评估流程。
- `evaluate_generation()` 对 test set 逐 batch 生成 `hypothesis`。
- `decode_sequence()` 把 token id 解码成文本。
- `evaluation.py::compute_report_metrics()` 用 `reference` 和 `hypothesis` 计算 BLEU / ROUGE-L。
- 如果打开 `--include-paper-metrics`，同一个 `reference` / `hypothesis` 会继续送入 `compute_paper_metrics()` 计算 METEOR / CIDEr。
- `write_report_outputs()` 写出 `metrics.json`、`predictions.jsonl`、`references.txt`、`hypotheses.txt`。

`predictions.jsonl` 每行主要字段：

```json
{
  "uid": "2",
  "history": "...",
  "reference": "...",
  "hypothesis": "...",
  "predicted_topics": []
}
```

## 6. Qwen 后处理评估

Qwen 后处理不是重新训练，也不是让 Qwen 看图。它只读取主模型已经生成的 `hypothesis`，再结合 `history` 做文本改写。

先准备 Qwen 环境。建议单独建环境，避免污染主模型 PyTorch 1.7 环境：

```bash
cd /root/autodl-tmp
python -m venv qwen-eval-env
source qwen-eval-env/bin/activate
pip install --upgrade pip
pip install transformers accelerate sentencepiece
```

假设 Qwen 模型目录是：

```text
/root/autodl-tmp/models/Qwen2.5-7B-Instruct/
```

运行：

```bash
cd /root/autodl-tmp/Context-Enhanced-Framework

python run_indiana_raw.py \
  --phase infer \
  --dataset-name indiana_raw \
  --data-root "/root/autodl-tmp/IU x-ray" \
  --output-dir "/root/autodl-tmp/Context-Enhanced-Framework/artifacts/iu_xray_raw" \
  --checkpoint-path "/root/autodl-tmp/Context-Enhanced-Framework/artifacts/iu_xray_raw/checkpoints/indiana_raw_Context_DenseNet121_MaxView2_NumLabel114_History.pt" \
  --num-workers 8 \
  --run-eval \
  --run-qwen-eval \
  --qwen-model-path "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
```

如果也要给 Qwen 后处理结果计算 METEOR / CIDEr，并且第 3 节依赖已经装好，加：

```bash
  --include-paper-metrics
```

这一步对应代码：

- `run_evaluation()` 先完成普通主模型评估，得到 `records`。
- `maybe_run_qwen()` 检查 `--run-qwen-eval` 和 `--qwen-model-path`。
- `qwen_postprocess.py::rewrite_reports_with_qwen()` 加载 Qwen，逐条改写 `hypothesis`。
- `qwen_postprocess.py::build_qwen_output_record()` 保存原始 `qwen_hypothesis`，并生成 `qwen_hypothesis_normalized`。
- `maybe_run_qwen()` 使用 `reference` 和 `qwen_hypothesis_normalized` 重新调用 `compute_report_metrics()`。
- `write_report_outputs(..., prefix="qwen")` 写出 Qwen 评估文件。

Qwen 输出文件：

```text
artifacts/iu_xray_raw/qwen_metrics.json
artifacts/iu_xray_raw/qwen_predictions.jsonl
artifacts/iu_xray_raw/qwen_references.txt
artifacts/iu_xray_raw/qwen_hypotheses.txt
```

注意：指标计算使用的是 `qwen_hypothesis_normalized`，不是原始自然语言格式的 `qwen_hypothesis`。这样可以避免大小写、标点粘连、空格格式导致 BLEU / ROUGE / METEOR / CIDEr 被额外惩罚。

## 7. 常用参数对应关系

| 参数 | 作用 | 对应代码 |
| --- | --- | --- |
| `--phase train` | 训练模型 | `main()` 里的训练循环 |
| `--phase infer` | 加载 checkpoint 后只推理/评估 | `utils.py::load()` + `run_evaluation()` |
| `--data-root` | Indiana 数据根目录 | `build_dataset_triplet()` 传给 `IndianaRawIUXRAY` |
| `--output-dir` | checkpoint 和评估文件输出目录 | `default_checkpoint_path()`、`write_report_outputs()` |
| `--checkpoint-path` | 指定要加载的模型权重 | `utils.py::load()` |
| `--pretrained-backbone` | 使用 torchvision 的 DenseNet121 预训练权重 | `build_model()` |
| `--run-eval` | 训练后或推理时输出 BLEU / ROUGE-L | `run_evaluation()` |
| `--include-paper-metrics` | 额外输出 METEOR / CIDEr | `compute_paper_metrics()` |
| `--run-qwen-eval` | 对主模型输出做 Qwen 后处理再评估 | `maybe_run_qwen()` |
| `--qwen-model-path` | 本地 Qwen 模型目录 | `rewrite_reports_with_qwen()` |

## 8. 常见问题

### checkpoint not found

检查 `--checkpoint-path` 是否真实存在：

```bash
ls /root/autodl-tmp/Context-Enhanced-Framework/artifacts/iu_xray_raw/checkpoints
```

### 缺少 pandas / sentencepiece / sklearn

安装主模型依赖：

```bash
pip install pandas pillow sentencepiece scikit-learn tqdm
```

### DenseNet 预训练权重下载失败

先去掉 `--pretrained-backbone` 跑通流程。这样会偏离论文设置，但能确认数据、训练、评估链路是否正常。

### METEOR / CIDEr 报错

如果命令里有 `--include-paper-metrics`，确认：

```bash
java -version
python -c "from pycocoevalcap.meteor.meteor import Meteor"
python -c "from pycocoevalcap.cider.cider import Cider"
```

如果暂时不需要论文扩展指标，去掉 `--include-paper-metrics`，基础 BLEU / ROUGE-L 不依赖这些包。

### Qwen 报 transformers 或 CUDA 错误

先确认主模型的 `metrics.json` 正常生成。Qwen 是额外后处理链路，可以等主模型评估稳定后再单独排查环境。

### 指标明显异常偏低

优先检查：

- `indiana_reports.csv` 里的 `findings` 是否正常读取。
- 图像文件名是否和 `indiana_projections.csv` 对得上。
- `qwen_metrics.json` 是否使用了 `qwen_hypothesis_normalized`。
- 是否误把没有规范化的 `qwen_hypothesis` 当作评估输入。
