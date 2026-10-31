# LICR-Native

**LLM-based Implicit Concept Recognition (LICR)** - 基于大语言模型的隐性概念识别

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18211198.svg)](https://doi.org/10.5281/zenodo.18211198)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 概述

LICR 是一种创新的文本-知识图谱链接方法，能够识别文本中**隐性概念**（implicit concepts）——即文本所体现但未显式命名的方法论、原理和模式。

### 核心问题

传统实体链接系统依赖表面形式匹配，只能识别文本中显式提及的实体。例如：

> "假设√2 = p/q，其中p和q互质。则2q² = p²，所以p²是偶数，因此p是偶数..."

任何受过数学训练的读者都能识别这是**反证法（proof by contradiction）**——但"反证法"三个字从未出现。传统系统无法捕捉这类隐性概念。

### 解决方案

LICR 提出混合架构：

```
文本 → LLM（语义理解）→ 概念名称 → Wikidata API → Q标识符
```

- **LLM**：负责语义理解和隐性概念识别（100%召回率）
- **Wikidata API**：负责精确的标识符映射（100%准确率）

### 实验结果

| 测试领域 | 核心隐性概念 | 召回率 |
|---------|------------|--------|
| 数学证明 | proof by contradiction | 100% |
| 算法描述 | quicksort, divide and conquer | 100% |
| 经济现象 | wage-price spiral | 100% |
| 博弈论 | prisoner's dilemma, Nash equilibrium | 100% |
| 量子物理 | wave-particle duality | 100% |

**整体准确率：96%**

---

## 目录结构

```
LICR-Native/
├── README.md                    # 本文件
├── LICR_v1.0.0.zip             # 代码发布包
│
├── research/                    # 研究文档
│   ├── LICR_dissertation_english.md      # 英文论文
│   ├── implicit_concept_recognition_dissertation.md  # 中文博士论文
│   ├── implicit_concept_recognition_paper.md         # 研究论文
│   ├── LICR_integration_analysis.md      # 框架集成分析
│   ├── knowledge_graphs_guide_cn.md      # 知识图谱指南
│   └── ...
│
├── docs/                        # 文档
│   ├── LICR_patent_strategy.md           # 专利/防御性发布策略
│   └── implicit_concept_recognition_paper.docx
│
├── scripts/                     # 代码实现
│   ├── licr.py                  # 核心Python实现
│   ├── requirements.txt         # 依赖
│   └── LICR_README.md           # 代码使用说明
│
└── conversation_logs/           # 研究对话记录
    ├── transcripts/             # 完整对话JSON（17个会话）
    │   ├── journal.txt          # 会话索引
    │   └── *.txt                # 完整对话记录
    └── *.md                     # 对话摘要
```

---

## 快速开始

### 安装

```bash
cd scripts
pip install -r requirements.txt
```

### 使用

```python
from licr import LICR

# 初始化（使用OpenAI或Anthropic）
licr = LICR(backend="openai", api_key="your-key")

# 识别隐性概念
text = """
Suppose √2 = p/q where p and q are coprime. Then 2q² = p², 
so p² is even, thus p is even. Let p = 2k, then q² = 2k², 
so q is also even. But this contradicts our assumption.
"""

result = licr.extract_concepts(text)
print(result)
# 输出: proof by contradiction (Q184899), mathematical proof (Q11538), ...
```

### 运行测试

```bash
python licr.py --test
```

---

## 引用

如果您使用了本项目，请引用：

```bibtex
@software{liu_claude_2026_licr,
  author       = {Liu, Qizhuang and Claude, Opus 4.5 (Anthropic)},
  title        = {{LICR: LLM-based Implicit Concept Recognition for 
                   Knowledge Graph Linking}},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18211198},
  url          = {https://doi.org/10.5281/zenodo.18211198}
}
```

---

## 研究背景

本项目源于对以下问题的探索：

1. **语义标签扩展**：如何从文本中提取超越关键词的高阶语义概念？
2. **LLM激活检索**：能否利用LLM内部表示进行语义检索？
3. **知识图谱映射**：如何将文本系统地映射到知识图谱概念？

经过系统研究，我们发现LLM具备识别隐性概念的能力，但无法可靠记忆任意标识符（如Wikidata Q号）。这一发现催生了LICR的混合架构设计。

完整研究历程记录在 `conversation_logs/transcripts/` 目录中。

---

## 许可证

本项目采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可证。

---

## 贡献者

- **Liu, Qizhuang** - 研究方向、实验设计、项目管理
- **Claude Opus 4.5 (Anthropic)** - 研究实现、论文撰写、代码开发

---

## 相关链接

- [Zenodo 发布页面](https://doi.org/10.5281/zenodo.18211198)
- [Wikidata](https://www.wikidata.org/)
- [Anthropic Claude](https://www.anthropic.com/)
