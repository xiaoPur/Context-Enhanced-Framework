# IU X-Ray 原始数据复现与 AutoDL 执行说明

## 1. 适用范围

这份说明对应当前仓库里的新入口脚本 `run_indiana_raw.py`。  
原始的 `train.py` 仍然保留在仓库中，但它是旧版硬编码入口，不建议再用于这次 Indiana 原始 CSV 复现。

当前改造支持的主流程是：

1. 直接读取原始 `images_normalized + indiana_projections.csv + indiana_reports.csv`
2. 使用 `findings` 作为生成目标
3. 优先复用仓库内置的 `iu_xray/file2label.json`
4. 训练完成后自动输出 `BLEU-1/2/3/4` 和 `ROUGE-L`
5. 可选用本地 `Qwen2.5-7B-Instruct` 对生成结果做后处理再评估

## 2. 服务器目录约定

你给出的目录结构可以直接使用：

- 项目目录：`/root/autodl-tmp/Context-Enhanced-framework/`
- 数据目录：`/root/autodl-tmp/IU x-ray/`
- 图像目录：`/root/autodl-tmp/IU x-ray/images_normalized/`
- 报告 CSV：`/root/autodl-tmp/IU x-ray/indiana_reports.csv`
- 投影 CSV：`/root/autodl-tmp/IU x-ray/indiana_projections.csv`

建议把训练输出单独放在项目目录下，例如：

- 输出目录：`/root/autodl-tmp/Context-Enhanced-framework/artifacts/iu_xray_raw/`

## 3. 默认镜像环境检查

进入项目后先确认基础环境：

```bash
cd /root/autodl-tmp/Context-Enhanced-Framework
python --version
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

建议在你当前 AutoDL 镜像里至少确认下面几项：

- Python 3.8
- PyTorch 1.7.x
- torchvision 0.8.x 左右
- CUDA 可用

## 4. 主模型依赖安装

如果默认镜像里缺少依赖，先补齐主模型所需包：

```bash
cd /root/autodl-tmp/Context-Enhanced-Framework
pip install pandas pillow sentencepiece scikit-learn tqdm
```

如果 `torchvision` 不可用，再补安装与当前 PyTorch 对应版本的 `torchvision`。  
如果镜像里已经自带，就不要重复升级，避免破坏默认 PyTorch 1.7 环境。

## 5. 训练命令

### 5.1 推荐训练命令

```bash
cd /root/autodl-tmp/Context-Enhanced-Framework

python run_indiana_raw.py \
  --phase train \
  --dataset-name indiana_raw \
  --data-root "/root/autodl-tmp/IU x-ray" \
  --output-dir "/root/autodl-tmp/Context-Enhanced-framework/artifacts/iu_xray_raw" \
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

这条命令会做几件事：

- 按 `uid` 构建 train / val / test 划分
- 自动把划分结果写到输出目录下的 `indiana_raw_splits.json`
- 训练模型并保存最佳 checkpoint
- 训练结束后直接在测试集上生成报告并计算指标

{
  "metrics": {
    "bleu_1": 0.466349,
    "bleu_2": 0.321742,
    "bleu_3": 0.236082,
    "bleu_4": 0.180148,
    "rouge_l": 0.376356
  },
  "qwen_metrics": null
}

### 5.2 如果服务器无法联网下载 DenseNet 预训练权重

去掉 `--pretrained-backbone`，先确认流程可跑通：

```bash
python run_indiana_raw.py \
  --phase train \
  --dataset-name indiana_raw \
  --data-root "/root/autodl-tmp/IU x-ray" \
  --output-dir "/root/autodl-tmp/Context-Enhanced-framework/artifacts/iu_xray_raw" \
  --batch-size 8 \
  --eval-batch-size 32 \
  --epochs 50 \
  --num-workers 8 \
  --run-eval
```

这会偏离论文设置，但能先验证训练链路是否通畅。

## 6. 只做推理与标准评估

如果你已经训练好了 checkpoint，可以单独做推理与指标计算：

```bash
cd /root/autodl-tmp/Context-Enhanced-framework

python run_indiana_raw.py \
  --phase infer \
  --dataset-name indiana_raw \
  --data-root "/root/autodl-tmp/IU x-ray" \
  --output-dir "/root/autodl-tmp/Context-Enhanced-framework/artifacts/iu_xray_raw" \
  --checkpoint-path "/root/autodl-tmp/Context-Enhanced-framework/artifacts/iu_xray_raw/checkpoints/indiana_raw_Context_DenseNet121_MaxView2_NumLabel114_History.pt" \
  --num-workers 8 \
  --run-eval
```

## 7. 标准评估输出文件

执行 `--run-eval` 后，输出目录里会有：

- `references.txt`
- `hypotheses.txt`
- `metrics.json`
- `predictions.jsonl`
- `indiana_raw_splits.json`

其中 `metrics.json` 至少包含：

- `bleu_1`
- `bleu_2`
- `bleu_3`
- `bleu_4`
- `rouge_l`

## 8. 标签文件说明

当前默认策略是：

- 优先使用仓库自带的 `iu_xray/file2label.json`
- key 映射规则为 `ecgen-radiology/{uid}.xml`
- 如果某个 `uid` 在旧标签文件中找不到，对应 14 维疾病标签会补零

这意味着：

- 如果你的原始 Indiana 数据与仓库自带 IU X-Ray 版本高度一致，通常可以直接跑
- 如果后续你补齐了外部标签文件，可以通过 `--external-label-file` 重新指定

## 9. Qwen2.5-7B-Instruct 后处理评估

### 9.1 先说明一个现实限制

主模型复现建议继续使用默认镜像里的 PyTorch 1.7 环境。  
Qwen2.5-7B-Instruct 更适合在单独的 Python 环境中安装 `transformers` 生态，而不是直接污染主复现环境。

也就是说：

- 主模型训练/评估：继续用默认镜像环境
- Qwen 后处理：建议单独建一个虚拟环境

### 9.2 创建可选的 Qwen 虚拟环境

```bash
cd /root/autodl-tmp
python -m venv qwen-eval-env
source qwen-eval-env/bin/activate
pip install --upgrade pip
pip install transformers accelerate sentencepiece
```

如果你打算在这个虚拟环境里直接跑 GPU 推理，需要再根据 AutoDL 当前驱动/CUDA 情况安装与你机器兼容的 PyTorch 版本。  
这一步不要覆盖主项目默认环境，尽量保持隔离。

### 9.3 Qwen 后处理评估命令

假设你已经把 `Qwen2.5-7B-Instruct` 下载到本地目录：

- `Qwen` 模型目录：`/root/autodl-tmp/models/Qwen2.5-7B-Instruct/`

那么可以在项目目录里执行：

```bash
cd /root/autodl-tmp/Context-Enhanced-Framework

python run_indiana_raw.py \
  --phase infer \
  --dataset-name indiana_raw \
  --data-root "/root/autodl-tmp/IU x-ray" \
  --output-dir "/root/autodl-tmp/Context-Enhanced-framework/artifacts/iu_xray_raw" \
  --checkpoint-path "/root/autodl-tmp/Context-Enhanced-framework/artifacts/iu_xray_raw/checkpoints/indiana_raw_Context_DenseNet121_MaxView2_NumLabel114_History.pt" \
  --num-workers 8 \
  --run-eval \
  --run-qwen-eval \
  --qwen-model-path "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
```

执行成功后会额外生成：

- `qwen_hypotheses.txt`
- `qwen_metrics.json`
- `qwen_predictions.jsonl`

## 10. 本地 Windows 阶段已经做过的验证

这次改造在本地无真实数据条件下已经做过：

- 原始 Indiana CSV 适配 mock 测试
- BLEU / ROUGE 指标函数测试
- Qwen prompt 构造测试
- `run_indiana_raw.py --help`
- 关键 Python 文件语法编译检查

还没有做的事情：

- 真实训练
- 真实 checkpoint 生成
- 真实 IU X-Ray 指标复现
- 真实 Qwen GPU 推理

这些都需要你上传到服务器后执行。

## 11. 常见问题排查

### 11.1 `checkpoint not found`

说明你在 `--phase infer` 时指定的 checkpoint 路径不对。  
先到输出目录确认：

```bash
ls /root/autodl-tmp/Context-Enhanced-framework/artifacts/iu_xray_raw/checkpoints
```

### 11.2 `No module named sentencepiece` / `pandas` / `sklearn`

补安装主模型依赖：

```bash
pip install pandas pillow sentencepiece scikit-learn tqdm
```

### 11.3 `torchvision` 下载预训练权重失败

先去掉 `--pretrained-backbone` 验证流程。  
如果一定要贴近论文设置，再单独处理权重下载或缓存。

### 11.4 `--run-qwen-eval` 报 transformers 相关错误

说明你还没在 Qwen 独立环境里装依赖，或者当前 PyTorch / CUDA 组合不兼容。  
优先保证主模型复现跑通，再单独排 Qwen 环境。

### 11.5 指标文件输出了，但结果很差

优先检查：

- 是否真的用了 `findings` 作为目标
- 图像文件名是否和 `indiana_projections.csv` 对得上
- `uid` 是否和旧标签文件映射成功
- 是否用到了预训练 DenseNet 权重

## 12. 最小执行顺序

如果你想最快开始跑，建议按这个顺序：

1. 上传当前项目到 `/root/autodl-tmp/Context-Enhanced-framework/`
2. 检查 Python / Torch / CUDA
3. 安装 `pandas pillow sentencepiece scikit-learn tqdm`
4. 先运行一条不带 Qwen 的训练命令
5. 确认 `metrics.json` 正常输出
6. 再决定是否单独搭 Qwen 后处理环境
