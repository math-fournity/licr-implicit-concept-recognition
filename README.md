# LICR：LLM 隐性概念识别研究档案

**LICR（LLM-based Implicit Concept Recognition）**：利用大语言模型识别文本中"未点名"的隐性概念（方法论、原理、模式），并链接到 Wikidata 知识库的混合架构研究。传统实体链接只能识别显式提到的实体；LICR 额外识别文本所体现但未明说的概念——例如从 √2 无理性的证明中识别出 `proof by contradiction (Q184899)`。

## 研究时间线

- **2026-01-10 ~ 01-12**：核心研究期——中英文论文、学位论文草稿、专利策略、`licr.py` v1.0.0 代码、Wikidata 链接实验、Zenodo DOI 发布准备。
- **2026-02-11**：LICR-Native 架构期——架构白皮书、与 LlamaIndex / LangChain / GraphRAG 的能力缺口对比（27 项缺口 / 四级风险）、完整问答记录汇编。

## 目录

| 目录 | 内容 |
|---|---|
| `paper/` | 中英文论文（含 docx） |
| `research/` | 文本→KG 完备标定、隐性概念链接、关键词/KG 指南、专利策略、集成分析等 |
| `code/` | `licr.py` v1.0.0（含原始 README、requirements.txt、.gitignore），CC BY 4.0 |
| `licr-native/` | LICR-Native 架构白皮书、能力缺口分析、系统架构 |
| `conversation-exports/` | Claude 对话完整导出（知识图谱标定 / Zenodo DOI 发布） |
| `transcripts/` | 研究期对话 transcript（文件名 = 时间戳 + 主题）及 journal 索引 |
| `archive-versions/` | 同一文件的历史版本 |

## 整理说明

- 本仓库由 AI 对话研究归档包整理而来，逐文件来源映射见 [PROVENANCE.md](PROVENANCE.md)。
- **同文异名**：`paper/Beyond Named Entities….md` 与多个压缩包中的 `LICR_dissertation_english.md` 字节相同（同一文档），去重后保留论文名；`research/超越命名实体….md` 与 `implicit_concept_recognition_dissertation.md` 同理。`LICR_README.md` 与 `code/README.md` 亦为同文。
- journal 索引中 `journal__16-cloud_backup_2026-01-12.txt` 混含微调研究条目，对应的微调 transcript 在姊妹仓库 `llm-finetuning-semantic-preservation`。
- 文件修改时间保留自原始资料；zip 不含创建时间，故创建时间与修改时间一致。
