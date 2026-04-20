# IU X-Ray 复现与评估流程说明

## 1. 本次结果是否正常

结论：这组新结果是正常的，而且比之前那组异常偏低的 Qwen 分数更可信。

本次结果如下：

| 指标 | 标准评估 `metrics` | Qwen 后处理评估 `qwen_metrics` | 差值 `qwen - base` |
| --- | ---: | ---: | ---: |
| BLEU-1 | 0.466462 | 0.455186 | -0.011276 |
| BLEU-2 | 0.321905 | 0.306694 | -0.015211 |
| BLEU-3 | 0.236257 | 0.220641 | -0.015616 |
| BLEU-4 | 0.180294 | 0.165368 | -0.014926 |
| ROUGE-L | 0.376459 | 0.376438 | -0.000021 |

这组结果可以这样理解：

1. 标准评估结果稳定，说明基础复现链路是正常的。
2. Qwen 后处理结果没有再出现之前那种“几乎腰斩”的异常下降，说明评估口径已经修正到合理状态。
3. Qwen 后处理后的 BLEU 略低、ROUGE-L 基本持平，这是合理现象。
4. 原因是当前 Qwen 的作用更像“报告语言润色器”，而不是一个直接看图生成报告的新模型。它会改写表述、压缩冗余、调整标点和句式，因此 n-gram 级别的 BLEU 往往不会显著提升，甚至可能略降；但只要整体内容保持一致，ROUGE-L 往往会保持稳定。

换句话说，当前结果更接近“Qwen 在不破坏主要语义的前提下做了轻量改写”，而不是“Qwen 把基础模型能力大幅增强”。

## 2. 为什么之前的 Qwen 分数会异常偏低

之前那组异常低分并不代表大模型真的把结果变差了很多，主要问题出在评估口径不一致。

旧版流程里：

1. 主模型输出的是分词后文本，形如 `lungs are clear .`
2. Qwen 输出的是自然语言句子，形如 `Lungs are clear.`
3. 评估函数使用的是非常直接的空格切分 `split()`，因此 `clear .` 和 `clear.` 会被当成不同 token
4. 同样，大小写变化、标点粘连、空格压缩都会被当成词面不一致

因此，旧版 `qwen_metrics` 实际上混合了两部分误差：

1. 真正的语义改写误差
2. 纯格式差异带来的额外惩罚

修正后流程会在 Qwen 评估前做规范化：

1. 转小写
2. 将常见标点拆成独立 token
3. 压缩多余空白

所以现在的 `qwen_metrics` 更接近 Qwen 改写后的真实内容差异，而不是格式差异。

## 3. 整个复现流程的实际数据流

下面按数据流说明这次 IU X-Ray 复现是怎么工作的。

### 3.1 数据集输入

当前复现使用的是 Indiana / IU X-Ray 原始 CSV 数据：

1. `indiana_reports.csv`
2. `indiana_projections.csv`
3. `images_normalized/`

在数据集构建阶段：

1. 从 `indiana_reports.csv` 读取每个 `uid` 的报告文本。
2. 将 `findings` 作为模型目标文本。
3. 将 `indication + comparison` 拼接成 `history`，作为文本上下文输入。
4. 从 `indiana_projections.csv` 找到该 `uid` 对应的影像文件，并在 `images_normalized/` 中加载图像。
5. 从 `iu_xray/file2label.json` 读取旧标签文件，得到基础疾病标签。
6. 再根据 `tools/count_nounphrase.json` 统计高频名词短语，为标签向量补充额外的 noun phrase 监督。

因此，单个样本的主要字段可以概括为：

- 图像：1 到 2 张胸片
- history：`indication + comparison`
- findings：监督目标
- label：旧标签 + noun phrase 标签扩展

### 3.2 数据集划分

数据集会按 `uid` 划分为：

1. train
2. val
3. test

如果没有现成的 split 文件，程序会按给定随机种子自动划分，并把划分结果写到 `indiana_raw_splits.json`。这样后续重复运行时可以复用同一划分。

### 3.3 主模型推理

当前主模型是一个“图像 + 文本历史 + 标签上下文”的多模态结构，核心流程如下：

1. 图像进入 CNN / MVCNN 分支，提取视觉特征。
2. `history` 进入文本编码分支，提取文本上下文特征。
3. 分类器分支融合图像与 history，得到每个 topic 对应的状态分数，也就是 `predicted_topics`。
4. 分类器分支再根据阈值 `threshold=0.15` 生成状态嵌入。
5. 生成器分支在这些上下文嵌入的条件下，自回归地生成目标报告，也就是 `hypothesis`。

因此，主模型在测试集上实际会同时产出两类东西：

1. `hypothesis`：生成的 findings 文本
2. `predicted_topics`：分类分支给出的 topic / state 分数

但是在当前实现里，真正用于标准文本评估的是 `hypothesis`，不是 `predicted_topics`。

## 4. 第一次评估：标准生成评估是怎么得到的

标准评估对应命令中的 `--run-eval`。

它的数据流是：

1. 加载 test 集。
2. 对每个样本执行主模型推理。
3. 生成 `hypothesis`。
4. 从 test 集真实标注中取出 `reference`，也就是 ground-truth findings。
5. 将 `reference` 和 `hypothesis` 成对送入 BLEU / ROUGE 计算函数。
6. 输出：
   - `references.txt`
   - `hypotheses.txt`
   - `metrics.json`
   - `predictions.jsonl`

这里的 `metrics.json` 就是你现在看到的：

```json
{
  "bleu_1": 0.466462,
  "bleu_2": 0.321905,
  "bleu_3": 0.236257,
  "bleu_4": 0.180294,
  "rouge_l": 0.376459
}
```

这一步没有用到大模型。

## 5. 第二次评估：Qwen 后处理评估是怎么得到的

Qwen 评估对应命令中的 `--run-qwen-eval`。

它不是重新训练，也不是让 Qwen 直接看图生成报告，而是在标准评估之后多做一步“文本后处理”。

### 5.1 当前实现中，大模型到底是怎么用的

当前实现里，Qwen 的输入不是图像，也不是 reference，而是：

1. `history`
2. 主模型已经生成好的 `hypothesis`

程序会把它们组织成一个 prompt，大意是：

1. 你是一个放射报告编辑器
2. 只能改写报告文本
3. 要尽量忠实于原草稿
4. 不要引入新发现
5. 不要删除非冗余发现
6. 返回简洁的 findings 文本

也就是说，Qwen 在当前版本里的角色是：

1. 一个后处理编辑器
2. 一个语言重写器
3. 一个“把主模型草稿写得更自然”的文本模块

它不是：

1. 一个直接看图的多模态模型
2. 一个和主模型端到端联合训练的模块
3. 一个显式读取 `predicted_topics` 再做诊断增强的模块

这一点在汇报时要特别讲清楚，因为这决定了 `qwen_metrics` 的解释方式。

### 5.2 Qwen 评估的数据流

Qwen 评估的完整链路是：

1. 先完成标准推理，拿到每条样本的 `reference`、`hypothesis`、`history`
2. 对每条样本构造 Qwen prompt
3. 让 Qwen 对 `hypothesis` 做一次报告改写
4. 得到原始改写结果 `qwen_hypothesis`
5. 再对 `qwen_hypothesis` 做规范化，得到 `qwen_hypothesis_normalized`
6. 用 `reference` 和 `qwen_hypothesis_normalized` 计算 BLEU / ROUGE
7. 输出：
   - `qwen_hypotheses.txt`
   - `qwen_metrics.json`
   - `qwen_predictions.jsonl`

这里要特别注意：

1. `qwen_hypotheses.txt` 是规范化后的评测文本，不是原始自然语言文本。
2. `qwen_predictions.jsonl` 同时保留原始 `qwen_hypothesis` 和 `qwen_hypothesis_normalized`。

因此，当前的 `qwen_metrics` 是“Qwen 改写后，再按和标准评估兼容的 token 口径重新计算”的结果。

## 6. 为什么这次 Qwen 指标比标准评估略低，但仍然正常

从结果上看，Qwen 后处理后：

1. BLEU-1 到 BLEU-4 都略低于标准评估
2. ROUGE-L 几乎不变

这可以这样解释：

1. Qwen 会把原始草稿中的重复表达、分词痕迹、标点形式改得更自然。
2. 这种改写会改变局部 n-gram，所以 BLEU 往往会小幅下降。
3. 但如果整体句意和主要发现没有明显改变，那么最长公共子序列仍然接近，因此 ROUGE-L 会比较稳定。

所以这组结果反而说明：

1. Qwen 没有像旧版那样被“格式失配”误伤
2. Qwen 主要做了语言层面的轻量编辑
3. 当前 Qwen 并没有带来显著的词面重合提升
4. 但它也没有破坏主要语义结构

这是一个合理且可信的现象。

## 7. 这次复现流程中最重要的汇报口径

向导师汇报时，建议使用下面这组口径。

### 7.1 可以明确汇报的内容

1. 已经完成 IU X-Ray 原始 CSV 数据链路的本地与服务器复现。
2. 主模型标准评估结果稳定，说明基础复现成功。
3. Qwen 模块当前作为后处理编辑器使用，不参与端到端训练。
4. 旧版 Qwen 评估低分主要来自格式失配，不是内容完全崩坏。
5. 修正评估口径后，Qwen 指标恢复到与标准评估接近的正常范围。

### 7.2 需要谨慎表述的内容

1. 不要把当前 `qwen_metrics` 解释为“大模型显著提升了生成质量”。
2. 更准确的说法是：大模型在当前实现中主要承担报告改写与语言整理功能。
3. 不要把当前实现直接等同于“一个新的端到端多模态报告生成模型”。

### 7.3 最稳妥的一句话总结

本次复现已经稳定得到主模型标准评估结果；大模型部分目前以“后处理报告编辑器”的方式接入，在修正评测口径后，其结果表现为对原始生成报告做轻量语言改写，整体内容保持稳定，但没有带来明显的词面指标提升。

## 8. 当前实现与论文式“大模型增强”的边界

从当前代码实现看，Qwen 的使用方式仍然偏保守。

当前版本的特点是：

1. Qwen 只接收 `history + hypothesis`
2. Qwen 不直接看图
3. Qwen 不直接读取 reference
4. Qwen 没有显式消费 `predicted_topics`

因此，当前的 Qwen 结果更接近“report polishing”而不是“diagnosis-guided generation enhancement”。

如果后续要继续推进，可以考虑两条方向：

1. 将分类分支的诊断结果显式整理成文本提示，再注入到 Qwen prompt 中
2. 增加临床一致性或医学事实层面的评估，而不只看 BLEU / ROUGE

## 9. 一次完整推理复现实验的操作命令

如果 checkpoint 已经训练好，那么当前推荐的复现实验命令是：

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

运行完成后重点查看：

1. `artifacts/iu_xray_raw/metrics.json`
2. `artifacts/iu_xray_raw/qwen_metrics.json`
3. `artifacts/iu_xray_raw/predictions.jsonl`
4. `artifacts/iu_xray_raw/qwen_predictions.jsonl`

## 10. 附注

如果后续需要和论文原始表格做严格逐项对齐，建议再补两件事：

1. 确认论文使用的原始评测栈与当前简化版 BLEU / ROUGE 实现是否完全一致
2. 区分“当前工程实现中的 Qwen 后处理”与“论文设想中的大模型增强方案”是否完全同口径

这一步主要是为了避免在汇报中把“工程实现结果”和“论文方法理想形态”混为一谈。
