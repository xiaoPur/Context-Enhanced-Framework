# IU X-Ray 原始数据适配与复现实验设计

## 目标

在本地 Windows 环境完成代码改造，并在没有本地真实数据集的前提下，把当前仓库整理成可直接上传到 AutoDL 服务器运行的版本。改造后的闭环包括：

1. 直接读取 Indiana / IU X-Ray 原始 `images_normalized + indiana_projections.csv + indiana_reports.csv`。
2. 以 `findings` 作为训练与评估目标文本。
3. 尽量保留论文主干的多模态上下文建模与标签分支。
4. 训练完成后自动评估 `BLEU-1/2/3/4` 与 `ROUGE-L`。
5. 支持使用本地下载好的 `Qwen2.5-7B-Instruct` 对主模型输出做后处理，再次评估。
6. 提供一份面向 AutoDL 默认镜像的中文执行说明书。

## 非目标

- 不在本地 Windows 上做真实训练或真实指标复现实验。
- 不引入大型实验管理框架或在线服务依赖。
- 不把 Qwen 后处理结果当成论文原始结果。

## 当前仓库问题

- `models.py` 中 `Generator.infer` 存在缩进错误，仓库当前状态无法通过语法检查。
- `train.py` 依赖大量硬编码配置，包括 GPU 编号、数据目录、训练阶段和输出位置。
- `datasets.py` 中的 `IUXRAY` 只能读取仓库自带的预处理 JSON/label 资源，无法直接读取你服务器上的原始 CSV 数据。
- 推理阶段只导出文本文件，不会自动计算生成指标。
- `prompt_enhancement/medical_report_processor.py` 依赖 OpenAI API，不适合作为本地离线 Qwen 评估链路。

## 设计原则

- 复现优先：尽量保留论文主干与现有网络结构。
- 最小侵入：优先新增适配层和工具模块，而不是大规模重写模型。
- 本地可验证：所有新增逻辑都要支持 mock 数据测试。
- 服务器可执行：脚本与说明书围绕 AutoDL 的 Ubuntu 18.04 / Python 3.8 / PyTorch 1.7.0 环境编写。

## 数据设计

### 输入目录

- 项目目录：`/root/autodl-tmp/Context-Enhanced-framework/`
- 数据根目录：`/root/autodl-tmp/IU x-ray/`
- 图像目录：`/root/autodl-tmp/IU x-ray/images_normalized/`
- 投影视图：`/root/autodl-tmp/IU x-ray/indiana_projections.csv`
- 报告数据：`/root/autodl-tmp/IU x-ray/indiana_reports.csv`

### 样本构造

- 以 `uid` 为病例主键。
- 从 `indiana_projections.csv` 按 `uid` 聚合同一病例的图像文件和投影方向。
- 从 `indiana_reports.csv` 读取 `indication`、`comparison`、`findings`、`impression`。
- `history` 使用 `indication + comparison`。
- `target` 只使用 `findings`。
- 每个病例最多取 2 张图像，保持与现有 `MVCNN` 输入接口一致。

### 标签策略

- 优先复用仓库内置的 [iu_xray/file2label.json](../../../../iu_xray/file2label.json)。
- 映射键采用 `ecgen-radiology/{uid}.xml`。
- 如果某个 `uid` 在旧标签文件中缺失，默认使用 14 维零标签并记录缺失 uid。
- 同时保留一个可选外部标签文件入口，便于你在服务器上补充 CheXpert 风格标签后重新跑实验。
- 最终标签维度维持为 `14 + 100 = 114`：14 维疾病标签加 100 维 noun phrase 额外标签。

### 数据切分

- 在 `uid` 级别进行 train/val/test 划分。
- 默认比例为 `0.7 / 0.1 / 0.2`，支持命令行覆盖。
- 支持把划分结果落盘为 JSON，避免多次运行导致切分漂移。

## 模型与训练设计

- 保留 `CNN + MVCNN + TNN + Classifier + Generator + Context` 主干。
- 保留标签分支和现有 `CELossTotal` 训练目标，以尽量贴近论文思路。
- 修复 `Generator.infer` 的结构性错误，并让生成路径支持稳定推理。
- 把训练入口改造成参数化 CLI，移除 `CUDA_VISIBLE_DEVICES="1"` 等写死配置。

## 评估设计

### 标准评估

训练结束或显式推理后，自动生成：

- `references.txt`
- `hypotheses.txt`
- `metrics.json`
- `predictions.jsonl`

统一计算：

- `BLEU-1`
- `BLEU-2`
- `BLEU-3`
- `BLEU-4`
- `ROUGE-L`

评估实现优先使用仓库内纯 Python 逻辑，避免强依赖 `nlg-eval`。

### Qwen 后处理评估

- 输入为主模型生成的 `findings`。
- 由本地 `Qwen2.5-7B-Instruct` 对生成文本做受控改写。
- 改写后重新计算同一套指标。
- 输出：
  - `qwen_hypotheses.txt`
  - `qwen_metrics.json`
  - `qwen_predictions.jsonl`

Qwen 链路单独做成可选脚本和可选依赖，避免影响主模型复现。

## 交付物

- 原始 Indiana CSV 数据适配代码
- 参数化训练/推理/评估入口
- 自动指标评估函数
- Qwen 后处理评估脚本
- AutoDL 中文执行说明书
- 本地 mock 数据测试
