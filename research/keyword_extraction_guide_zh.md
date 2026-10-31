# Mac 本地语义关键词提取实用技术指南

## 核心结论

在 Mac Studio 64GB 上使用 Qwen3-0.6B 等小模型提取最大化关键词（包括隐含和潜在概念），**最佳方案是组合多个互补工具**：
- **KeyBERT**：基于嵌入的关键词提取
- **BERTopic**：潜在主题发现
- **GLiNER**：零样本实体识别
- **小型 LLM**：关键词扩展和生成

所有工具都能在 Apple Silicon 上高效运行（MLX 或 llama.cpp 后端），总内存需求低于 15GB。

**关键洞察**：没有单一工具能覆盖所有关键词提取场景。混合流水线结合各工具优势，才能实现最大关键词覆盖。

---

## 工具分类对比

| 类型 | 工具 | 优势 | 局限 | 速度 |
|------|------|------|------|------|
| **统计方法** | YAKE, RAKE | 快速，无需模型 | 无法捕捉语义关系 | ~10ms/千词 |
| **嵌入方法** | KeyBERT | 语义相似度 | 只能提取原文词汇 | ~200ms/文档 |
| **主题模型** | BERTopic, Top2Vec | 发现潜在主题 | 需要语料库级别操作 | 一次性训练 |
| **生成方法** | LLM, T5 | 可生成原文没有的词 | 较慢 | ~1-5s/文档 |

---

## 第一层：统计关键词提取

### YAKE（最快）

```bash
pip install yake
```

```python
import yake

kw_extractor = yake.KeywordExtractor(
    lan="zh",           # 支持中文
    n=3,                # 最大词组长度
    top=20              # 返回前20个关键词
)
keywords = kw_extractor.extract_keywords(text)
```

**特点**：
- 速度：~10ms / 1000词
- 内存：< 50MB
- 无需外部模型
- 局限："机器学习"和"ML"被视为无关词

---

## 第二层：语义关键词提取

### KeyBERT（推荐）

```bash
pip install keybert sentence-transformers
```

```python
from keybert import KeyBERT

# 使用轻量嵌入模型
kw_model = KeyBERT(model='all-MiniLM-L6-v2')  # 90MB

keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(1, 3),
    stop_words='english',
    use_mmr=True,        # 最大边际相关性，减少冗余
    diversity=0.7,
    top_n=20
)
```

**嵌入模型选择**：

| 模型 | 大小 | 质量 | 推荐场景 |
|------|------|------|----------|
| all-MiniLM-L6-v2 | 90MB | ⭐⭐⭐ | 通用场景 |
| BGE-small-zh | 130MB | ⭐⭐⭐⭐ | 中文优化 |
| BGE-M3 | 2GB | ⭐⭐⭐⭐⭐ | 最高质量 |

---

## 第三层：潜在主题发现

### BERTopic（发现隐含概念的关键）

```bash
pip install bertopic
```

```python
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# 使用 KeyBERTInspired 提升关键词质量
representation_model = KeyBERTInspired()

topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    representation_model=representation_model,
    calculate_probabilities=True,  # 软主题分配
    verbose=True
)

# 训练（需要文档集合）
topics, probs = topic_model.fit_transform(documents)

# 获取每个文档的主题关键词
doc_info = topic_model.get_document_info(documents)
```

**BERTopic 工作流程**：
```
文档 → 嵌入(Sentence Transformer) → 降维(UMAP) → 聚类(HDBSCAN) → c-TF-IDF → 主题关键词
```

**为什么 BERTopic 能发现隐含概念**：
- 单文档方法只能看到一篇文档
- 主题模型看到整个语料库的模式
- 能发现跨文档的语义结构

---

## 第四层：零样本实体提取

### GLiNER（自定义实体类型）

```bash
pip install gliner
```

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# 自定义你想提取的实体类型
labels = [
    "技术", "算法", "框架", "概念", 
    "方法论", "工具", "产品", "公司"
]

entities = model.predict_entities(
    text, 
    labels, 
    threshold=0.5
)

# 输出: [{"text": "深度学习", "label": "技术", "score": 0.92}, ...]
```

**GLiNER 优势**：
- 零样本：无需训练，运行时指定实体类型
- 速度：~100ms/文档，比 LLM 快得多
- 质量：匹配 GPT-4 的零样本 NER 性能

---

## 第五层：LLM 关键词扩展

### 使用 Qwen3-0.6B 生成隐含关键词

```python
import requests

def expand_keywords(explicit_keywords, context):
    prompt = f"""基于以下从文档中提取的关键词：
{explicit_keywords}

以及文档上下文：
{context[:500]}

请生成 10 个相关但未在原文中出现的概念、主题或关键词："""

    response = requests.post(
        "http://localhost:8080/completion",
        json={
            "prompt": prompt,
            "n_predict": 200,
            "temperature": 0.7
        }
    )
    return response.json()["content"]
```

---

## 完整混合流水线

```python
from keybert import KeyBERT
from gliner import GLiNER
import yake

class HybridKeywordExtractor:
    def __init__(self):
        # 初始化所有模型
        self.yake = yake.KeywordExtractor(lan="zh", n=3, top=30)
        self.keybert = KeyBERT(model='all-MiniLM-L6-v2')
        self.gliner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        self.entity_labels = ["技术", "概念", "方法", "工具", "产品"]
        
    def extract(self, text, use_llm=True):
        results = {
            "statistical": [],
            "semantic": [],
            "entities": [],
            "implicit": []
        }
        
        # 1. 统计关键词 (YAKE)
        yake_kws = self.yake.extract_keywords(text)
        results["statistical"] = [kw for kw, score in yake_kws]
        
        # 2. 语义关键词 (KeyBERT)
        bert_kws = self.keybert.extract_keywords(
            text, use_mmr=True, diversity=0.7, top_n=20
        )
        results["semantic"] = [kw for kw, score in bert_kws]
        
        # 3. 实体 (GLiNER)
        entities = self.gliner.predict_entities(
            text, self.entity_labels, threshold=0.5
        )
        results["entities"] = [e["text"] for e in entities]
        
        # 4. 隐含关键词 (LLM) - 可选
        if use_llm:
            explicit = results["statistical"] + results["semantic"]
            results["implicit"] = self._llm_expand(explicit, text)
        
        # 5. 去重合并
        all_keywords = self._merge_and_dedupe(results)
        return all_keywords
    
    def _llm_expand(self, keywords, context):
        # 调用本地 LLM
        pass
    
    def _merge_and_dedupe(self, results):
        # 基于嵌入相似度去重
        pass
```

---

## 内存配置建议

### 64GB Mac Studio 推荐配置

| 配置级别 | 嵌入模型 | LLM | 总内存 | 适用场景 |
|----------|----------|-----|--------|----------|
| **轻量** | MiniLM (90MB) | Qwen3-0.6B Q4 (400MB) | ~3GB | 高吞吐量处理 |
| **均衡** ✅ | BGE-small (130MB) | Qwen3-1.7B Q4 (1GB) | ~5GB | 质量与速度平衡 |
| **高质量** | BGE-base (440MB) | Qwen3-4B Q4 (2.5GB) | ~8GB | 最佳小模型质量 |
| **最大** | BGE-M3 (2GB) | Qwen3-8B Q4 (5GB) | ~12GB | 接近商业级质量 |

**均衡配置留出 55GB+ 空间**用于文档处理、BERTopic 训练和系统开销。

---

## 处理你的 600 个文件

### 推荐流程

```
600 个文件 (各 1M tokens)
         │
         ▼
┌─────────────────────────────────────┐
│  第一步：分块处理                      │
│  每个文件 → 32K chunks × 32          │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  第二步：并行提取（每个 chunk）         │
│  • YAKE: 统计关键词                   │
│  • KeyBERT: 语义关键词                │
│  • GLiNER: 实体                      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  第三步：聚合 + 主题发现               │
│  • 合并 chunk 关键词                  │
│  • BERTopic 发现潜在主题              │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  第四步：LLM 扩展（可选）              │
│  • 基于已提取关键词生成隐含概念         │
└─────────────────────────────────────┘
```

### 时间估算

| 步骤 | 单文件时间 | 600文件总时间 |
|------|-----------|--------------|
| 分块 | ~10s | ~1.7h |
| YAKE | ~30s | ~5h |
| KeyBERT | ~2min | ~20h |
| GLiNER | ~1min | ~10h |
| BERTopic | 一次性训练 | ~2-4h |
| LLM扩展 | ~5min | ~50h |

**并行处理后总时间**：约 15-20 小时（不含 LLM 扩展）

---

## 安装命令汇总

```bash
# 基础工具
pip install yake keybert sentence-transformers gliner bertopic

# Apple Silicon 优化
pip install mlx mlx-lm

# 或者用 llama.cpp
brew install llama.cpp
```

---

## 总结

| 需求 | 推荐工具 |
|------|----------|
| 原文关键词（快速） | YAKE |
| 原文关键词（语义） | KeyBERT |
| 命名实体 | GLiNER |
| 潜在主题/隐含概念 | **BERTopic** |
| 生成原文没有的关键词 | **Qwen3 + 提示词** |

**核心建议**：BERTopic 是发现隐含概念的关键。单文档方法无法发现跨文档的语义结构，必须用主题模型在语料库级别进行分析。
