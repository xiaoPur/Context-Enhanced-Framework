# IU X-Ray 复现与评估结果报告

## 1. 本次复现结果

本次复现已经完成主模型标准评估、Qwen 后处理评估，以及论文口径扩展指标 `METEOR` / `CIDEr` 的重新计算。

当前结果来自：

- `artifacts/iu_xray_raw/metrics.json`
- `artifacts/iu_xray_raw/qwen_metrics.json`

结果如下：

| 指标 | 标准评估 `metrics` | Qwen 后处理评估 `qwen_metrics` | 差值 `qwen - base` |
| --- | ---: | ---: | ---: |
| BLEU-1 | 0.457703 | 0.466005 | +0.008302 |
| BLEU-2 | 0.319700 | 0.318358 | -0.001342 |
| BLEU-3 | 0.237431 | 0.232374 | -0.005057 |
| BLEU-4 | 0.183604 | 0.177303 | -0.006301 |
| ROUGE-L | 0.372812 | 0.372822 | +0.000010 |
| METEOR | 0.209366 | 0.197922 | -0.011444 |
| CIDEr | 0.420932 | 0.458369 | +0.037437 |

## 2. 结果解读

这组结果整体是正常的。标准评估和 Qwen 后处理评估没有出现明显断崖式差异，说明当前评估链路已经比旧版“直接拿自然语言 Qwen 输出算指标”的方式更稳定。

从指标变化看：

1. `BLEU-1` 在 Qwen 后处理后略有提升，说明单词级别重合略好。
2. `BLEU-2/3/4` 略有下降，说明 Qwen 改写改变了部分连续 n-gram 结构。
3. `ROUGE-L` 几乎不变，说明整体最长公共子序列和主要内容顺序基本稳定。
4. `METEOR` 略降，说明 Qwen 的改写没有在 METEOR 口径下带来更高的词形、匹配或顺序收益。
5. `CIDEr` 提升，说明 Qwen 后处理后的文本在部分具有区分度的 n-gram 上更接近参考报告。

因此，当前 Qwen 的效果不能简单表述为“全面提升”或“明显变差”。更准确的结论是：Qwen 主要改变了表达方式，使部分词面和 CIDEr 口径受益，但也破坏了一些高阶连续 n-gram 和 METEOR 匹配。

## 3. 复现流程的数据来源

当前复现使用 Indiana / IU X-Ray 原始 CSV 数据：

1. `indiana_reports.csv`
2. `indiana_projections.csv`
3. `images_normalized/`

数据构建逻辑如下：

1. 从 `indiana_reports.csv` 读取每个 `uid` 的报告文本。
2. 使用 `findings` 作为报告生成目标。
3. 将 `indication + comparison` 拼接成 `history`，作为文本上下文输入。
4. 从 `indiana_projections.csv` 找到对应图像文件，并在 `images_normalized/` 中加载胸片。
5. 优先使用仓库内置的 `iu_xray/file2label.json` 作为疾病标签来源。
6. 使用 `tools/count_nounphrase.json` 补充 noun phrase 标签监督。

单个样本主要包含：

- 图像：1 到 2 张胸片
- `history`：检查指征和对比信息
- `reference`：真实 findings
- `hypothesis`：主模型生成的 findings
- `predicted_topics`：分类分支输出的 topic / state 分数

## 4. 标准生成评估

标准评估对应命令中的 `--run-eval`。

标准评估流程是：

1. 加载 test 集。
2. 使用主模型对每个样本生成报告。
3. 从 test 集真实标注中取出 `reference`。
4. 从模型输出中取出 `hypothesis`。
5. 使用 `reference` 和 `hypothesis` 计算指标。
6. 输出 `metrics.json`、`predictions.jsonl`、`references.txt`、`hypotheses.txt`。

当前 `metrics.json` 为：

```json
{
  "bleu_1": 0.457703,
  "bleu_2": 0.3197,
  "bleu_3": 0.237431,
  "bleu_4": 0.183604,
  "rouge_l": 0.372812,
  "meteor": 0.209366,
  "cider": 0.420932
}
```

其中：

- BLEU / ROUGE-L 由仓库内 `evaluation.py::compute_report_metrics()` 计算。
- METEOR / CIDEr 在命令加入 `--include-paper-metrics` 后，由 `evaluation.py::compute_paper_metrics()` 调用 `pycocoevalcap` 计算。

这一步没有使用 Qwen。

## 5. Qwen 后处理评估

Qwen 后处理评估对应命令中的 `--run-qwen-eval`。

当前实现中，Qwen 的输入不是图像，也不是 reference，而是：

1. `history`
2. 主模型已经生成好的 `hypothesis`

Qwen 的角色是报告后处理编辑器：它对主模型生成的草稿做语言层面的整理和改写。它不参与主模型训练，也不直接看图。

从本次 `qwen_predictions.jsonl` 保存的 `qwen_prompt` 看，本次 artifacts 对应的是旧版“报告编辑器”提示词。该提示词强调保持临床忠实、不要引入新发现、不要删除非冗余发现，并要求输出简洁报告文本。因此这组结果应解释为“旧版 Qwen 润色型提示词”的评估结果，而不是后续更保守的“最小修改型提示词”的结果。

Qwen 评估流程是：

1. 先完成标准主模型推理，得到每条样本的 `reference`、`hypothesis`、`history`。
2. 将 `history + hypothesis` 组织成 Qwen prompt。
3. 让 Qwen 对 `hypothesis` 做报告改写，得到 `qwen_hypothesis`。
4. 对 `qwen_hypothesis` 做评估规范化，得到 `qwen_hypothesis_normalized`。
5. 使用 `reference` 和 `qwen_hypothesis_normalized` 计算 BLEU / ROUGE-L / METEOR / CIDEr。
6. 输出 `qwen_metrics.json`、`qwen_predictions.jsonl`、`qwen_references.txt`、`qwen_hypotheses.txt`。

当前 `qwen_metrics.json` 为：

```json
{
  "bleu_1": 0.466005,
  "bleu_2": 0.318358,
  "bleu_3": 0.232374,
  "bleu_4": 0.177303,
  "rouge_l": 0.372822,
  "meteor": 0.197922,
  "cider": 0.458369
}
```

这里最重要的一点是：Qwen 评估没有使用原始自然语言格式的 `qwen_hypothesis` 直接算分，而是使用规范化后的 `qwen_hypothesis_normalized`。这样可以避免大小写、标点粘连、空格差异带来的额外惩罚。

## 6. 为什么 Qwen 结果是混合变化

当前 Qwen 后处理并没有带来所有指标的同步提升，这符合本次旧版润色型提示词的使用方式。

旧版提示词鼓励 Qwen 做一定程度的文本整理和简化，因此它可能：

1. 将局部词语变得更自然，使 `BLEU-1` 上升。
2. 改变短语顺序或句式，使高阶 BLEU 下降。
3. 保持整体内容顺序，使 `ROUGE-L` 基本不变。
4. 改写部分医学表达，使 `METEOR` 的匹配收益下降。
5. 保留或强化某些关键 n-gram，使 `CIDEr` 上升。

因此，当前结果更适合解释为“润色型语言后处理带来的指标结构变化”，而不是“Qwen 作为新模型显著提升生成质量”。

## 7. 可用于汇报的结论

可以明确汇报：

1. 已经完成 IU X-Ray 原始 CSV 数据链路的训练、推理和评估复现。
2. 标准评估已经输出 BLEU-1/2/3/4、ROUGE-L、METEOR、CIDEr。
3. Qwen 后处理评估也已经使用同一套指标重新计算。
4. Qwen 后处理结果与主模型结果整体接近，没有出现旧版格式失配导致的异常低分。
5. Qwen 的影响是混合的：BLEU-1 和 CIDEr 提升，BLEU-2/3/4 与 METEOR 略降，ROUGE-L 基本不变。

需要谨慎表述：

1. 不要把当前 Qwen 结果说成端到端多模态模型能力提升。
2. 不要说 Qwen 全面提升了报告生成质量。
3. 更准确的说法是：Qwen 当前作为后处理编辑器，主要改变语言表达和局部 n-gram 分布。

最稳妥的一句话总结：

> 本次复现已经完成主模型和 Qwen 后处理两套评估，并补充了 METEOR / CIDEr 指标；结果显示 Qwen 后处理没有破坏整体内容结构，但其收益主要体现在 BLEU-1 和 CIDEr 上，高阶 BLEU 与 METEOR 略有下降，因此当前 Qwen 更适合作为报告语言后处理模块来解释。

## 8. 当前实现与论文式增强的边界

本次 artifacts 对应的 Qwen 使用方式仍然是后处理式的：

1. Qwen 只接收 `history + hypothesis`。
2. Qwen 不直接看 X-Ray 图像。
3. Qwen 不读取 reference。
4. Qwen 没有显式使用 `predicted_topics`。
5. Qwen 不参与端到端训练。

因此，本次实验更接近 report polishing，而不是 diagnosis-guided generation enhancement。

另外，代码后续可以使用更保守的“最小修改型”提示词重新跑一轮 Qwen 评估。那一轮如果重新生成 artifacts，需要单独记录为新的实验结果，不能和本报告中的旧版 prompt 结果混在一起比较。

如果后续要进一步贴近“大模型增强”表述，可以考虑：

1. 将分类分支输出的诊断主题显式整理成文本，加入 Qwen prompt。
2. 增加医学事实一致性评估，而不只依赖词面指标。
3. 对 Qwen 改写前后的病例进行人工抽样审查，确认有没有引入或删除临床发现。

## 9. 本次复现实验命令

如果 checkpoint 已经训练好，当前推荐的完整评估命令是：

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
  --include-paper-metrics \
  --run-qwen-eval \
  --qwen-model-path "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
```

运行完成后重点查看：

1. `artifacts/iu_xray_raw/metrics.json`
2. `artifacts/iu_xray_raw/qwen_metrics.json`
3. `artifacts/iu_xray_raw/predictions.jsonl`
4. `artifacts/iu_xray_raw/qwen_predictions.jsonl`

## 10. 附注

本报告中的 METEOR / CIDEr 来自服务器端扩展评估依赖。后续如果更换评估包或分词口径，分数可能会发生变化。汇报时建议同时注明：

1. 使用 `reference` 与 `hypothesis` 计算标准评估。
2. 使用 `reference` 与 `qwen_hypothesis_normalized` 计算 Qwen 后处理评估。
3. METEOR / CIDEr 通过 `--include-paper-metrics` 启用。
4. 本次 Qwen 结果对应 `qwen_predictions.jsonl` 中保存的旧版润色型 prompt；若换成最小修改型 prompt，需要重新跑并另行记录。
