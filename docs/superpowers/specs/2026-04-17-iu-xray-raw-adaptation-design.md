# IU X-Ray 原始数据适配与快速复现实验设计

## 目标

在本地 Windows 环境完成代码改造，并在不依赖本地真实数据的前提下，把当前仓库改造成可直接上传到 AutoDL 服务器运行的最短闭环版本。目标闭环如下：

1. 直接读取原始 Indiana / IU X-Ray CSV 与 PNG 数据。
2. 仅以 `findings` 作为文本生成目标。
3. 保留论文主线中的图像与文本上下文建模思路，但移除显式分类监督依赖。
4. 训练完成后直接输出 `BLEU-1/2/3/4` 与 `ROUGE-L`。
5. 额外支持使用本地下载好的 `Qwen2.5-7B-Instruct` 对模型生成结果做后处理，再次评估。
6. 提供一份适配 AutoDL 默认镜像的中文执行说明书。

## 非目标

- 不追求对论文全部监督信号的严格 1:1 复现。
- 不接入额外的医学分类标签提取流程。
- 不在本地 Windows 环境进行真实训练或真实指标验证。
- 不引入复杂的实验管理框架、服务化推理框架或额外在线依赖。

## 当前仓库问题

当前仓库存在以下会阻塞快速运行的问题：

- `models.py` 中 `Generator.infer` 存在缩进错误，原始代码无法通过语法检查。
- `train.py` 使用全局硬编码配置，包含固定 GPU 号、固定数据目录、固定阶段切换与固定输出逻辑。
- `datasets.py` 中的 `IUXRAY` 依赖仓库内预处理好的 JSON / label 文件，不兼容服务器上的原始 CSV 数据布局。
- 评估流程仅导出文本文件，不会自动计算常用生成指标。
- `prompt_enhancement/medical_report_processor.py` 依赖旧 OpenAI API，不适合本地离线 Qwen 推理。

## 设计原则

- 最短闭环优先：先让训练、推理、评估、Qwen 后处理都能跑通。
- 最小侵入优先：尽量保留原仓库的模型组织方式，不做大规模重构。
- 本地无数据可验证：所有新增逻辑都要支持用伪样本做 smoke test。
- 服务端直接可执行：说明书与脚本默认围绕 AutoDL 提供的 Ubuntu 18.04 / PyTorch 1.7.0 / Python 3.8 环境编写。

## 数据设计

### 输入目录

服务器上使用如下数据目录：

- 项目目录：`/root/autodl-tmp/Context-Enhanced-framework/`
- 数据目录：`/root/autodl-tmp/IU x-ray/`
- 图像目录：`/root/autodl-tmp/IU x-ray/images_normalized/`
- 投影视图文件：`/root/autodl-tmp/IU x-ray/indiana_projections.csv`
- 报告文件：`/root/autodl-tmp/IU x-ray/indiana_reports.csv`

### 样本构造

- 以 `uid` 为病例主键。
- 从 `indiana_projections.csv` 中取该 `uid` 关联的图像文件名与投影方向。
- 从 `indiana_reports.csv` 中取该 `uid` 对应报告字段。
- `history` 由 `indication` 和 `comparison` 拼接得到；缺失值按空字符串处理。
- 训练目标只使用 `findings`；缺失或空文本样本被过滤。
- 每个病例最多取 2 张图，保持与原仓库多视图输入接口一致。

### 数据切分

- 默认在数据集类中完成 train / val / test 切分。
- 使用固定随机种子，默认比例为 `0.7 / 0.1 / 0.2`。
- 本地无数据阶段允许通过小型 mock CSV + mock PNG 构造最小可运行样本。

### 词表策略

- 默认直接复用仓库自带 `iu_xray/nlmcxr_unigram_1000.model`。
- 不新增词表训练流程，避免增加额外依赖与准备成本。

## 模型与训练设计

### 主体策略

- 保留 `CNN + MVCNN + TNN + Context + Generator` 这一套主干。
- 去除训练阶段对显式 `label` 输入和 `CELossTotal` 分类分支损失的依赖。
- 将训练目标收缩为单一文本生成损失。

### 代码层落地

- 修复 `models.py` 中 `Generator.infer` 的结构性错误。
- 让 `Context.forward` 支持无标签条件下的生成路径。
- 简化 `Classifier` / `Context` 之间的调用关系，确保在没有外部标签文件时仍能产出稳定的上下文表示。

### 训练入口

新增可配置训练入口，至少支持以下参数：

- 数据根目录
- 输出目录
- 训练阶段或推理阶段
- batch size
- epoch 数
- num_workers
- 随机种子
- device
- checkpoint 路径
- 是否执行标准评估
- 是否执行 Qwen 后处理评估
- Qwen 模型目录

## 评估设计

### 标准评估

训练结束或显式推理后，自动导出：

- `predictions.txt`
- `references.txt`
- `metrics.json`

指标统一为：

- `BLEU-1`
- `BLEU-2`
- `BLEU-3`
- `BLEU-4`
- `ROUGE-L`

优先使用本地 Python 依赖完成实现，避免强依赖 `nlg-eval`。

### Qwen 后处理评估

- 输入为模型原始生成的 `findings` 文本。
- 使用本地 `Qwen2.5-7B-Instruct` 进行简洁报告改写。
- 改写目标是让文本更通顺、结构更自然，但不引入与原结果无关的新病灶。
- 输出：
  - `qwen_predictions.txt`
  - `qwen_metrics.json`

该链路只做文本后处理，不做视觉输入推理，以保证最快落地。

## 文档设计

补充中文执行说明书，内容包含：

- AutoDL 默认镜像下的环境检查与依赖安装
- 数据目录结构说明
- 训练命令
- 标准评估命令
- Qwen 后处理评估命令
- 常见报错与排查建议

## 本地验证策略

由于本地没有真实数据，本轮交付只承诺以下验证：

- Python 语法与模块导入检查
- mock 数据集 smoke test
- 单步前向与评估脚本 dry-run

真实训练与真实指标验证由用户上传服务器后执行。

## 风险与取舍

- 去掉标签监督后，结果不应被描述为论文完整原实验复现，而应表述为“面向文本生成效果的简化复现版本”。
- 复用现有 SentencePiece 词表能加快启动，但对你的数据分布未必最优。
- Qwen 后处理会带来额外 GPU 显存占用，但对 48GB 显存单卡是可接受方案。

## 交付物

- 改造后的训练与数据读取代码
- 标准评估函数
- Qwen 后处理评估函数
- 中文执行说明书
- 可在本地无数据环境下完成的基础验证脚本
