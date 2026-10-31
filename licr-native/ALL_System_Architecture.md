# ALL: Augmented LLM with LICR
## 基于隐性概念识别的智能知识库系统架构设计

**版本**: 1.0  
**日期**: 2026-01-12  
**状态**: 架构设计

---

## 一、系统概述

### 1.1 系统定位

ALL（Augmented LLM with LICR）是一个基于大语言模型的智能知识库系统，其核心创新在于利用LICR（LLM-based Implicit Concept Recognition）技术，实现对知识库中**隐性概念**的识别与利用。

与传统RAG系统仅能进行关键词/语义匹配不同，ALL能够识别文本中**隐含但未明示的方法论、原理和模式**，从而实现真正的"理解式检索"。

### 1.2 核心能力

| 能力层级 | 传统RAG | ALL系统 |
|---------|--------|---------|
| 显性概念识别 | ✅ 关键词匹配 | ✅ 语义匹配 |
| 隐性概念识别 | ❌ 无法识别 | ✅ LICR识别 |
| 跨表述泛化 | ❌ 依赖表层相似 | ✅ ICL推理判断 |
| 知识推理 | ⚠️ 简单拼接 | ✅ 多层推理整合 |

### 1.3 设计原则

基于深度技术调研的关键发现，ALL架构遵循以下原则：

1. **ICL优于微调泛化**：利用模型的In-Context Learning能力进行概念匹配，避免Reversal Curse问题
2. **级联处理架构**：小模型快速筛选 + 大模型深度分析，平衡效率与准确性
3. **原子化概念表示**：将复杂DSL概念分解为公共概念的组合，提高识别泛化性
4. **显性定义驱动**：所有概念以自然语言定义形式存在，支持动态更新

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ALL System                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   Input     │    │              Knowledge Base (KB)                  │   │
│  │             │    │  ┌────────────────┐  ┌─────────────────────────┐ │   │
│  │ • 新文本块  │    │  │ Concept Store  │  │    Document Store       │ │   │
│  │ • 用户提问  │    │  │                │  │                         │ │   │
│  │             │    │  │ • DSL定义      │  │ • 原始文档              │ │   │
│  └──────┬──────┘    │  │ • 原子分解     │  │ • 文本块                │ │   │
│         │           │  │ • 示例集合     │  │ • 概念标注              │ │   │
│         ▼           │  └────────────────┘  └─────────────────────────┘ │   │
│  ┌──────────────┐   └──────────────────────────────────────────────────┘   │
│  │  Embedding   │                          │                                │
│  │   Layer      │◄─────────────────────────┘                                │
│  │  (预筛选)    │                                                           │
│  └──────┬───────┘                                                           │
│         │ Top-K 候选概念                                                     │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    LICR Recognition Engine                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                     LLM-C (Fast Judge)                           │ │  │
│  │  │                                                                   │ │  │
│  │  │   Input: [概念定义] + [文本块]                                    │ │  │
│  │  │   Task: 判断文本是否体现该概念                                    │ │  │
│  │  │   Output: {concept_id, confidence, evidence}                     │ │  │
│  │  │                                                                   │ │  │
│  │  │   Models: Qwen2.5-1.5B / Phi-3-mini / DeBERTa-v3                 │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                        │  │
│  │              ┌───────────────┴───────────────┐                       │  │
│  │              ▼                               ▼                        │  │
│  │     高置信度结果                      低置信度/边界案例               │  │
│  │     (confidence > 0.8)               (0.3 < confidence < 0.8)        │  │
│  │              │                               │                        │  │
│  │              │                               ▼                        │  │
│  │              │                    ┌─────────────────────┐            │  │
│  │              │                    │  LLM-B (Deep Judge) │            │  │
│  │              │                    │                     │            │  │
│  │              │                    │  详细推理分析        │            │  │
│  │              │                    │  原子级别匹配        │            │  │
│  │              │                    │  多概念关联分析      │            │  │
│  │              │                    └──────────┬──────────┘            │  │
│  │              │                               │                        │  │
│  │              └───────────────┬───────────────┘                       │  │
│  │                              ▼                                        │  │
│  │                    Recognition Results                                │  │
│  │                    {concept_matches[], evidence[], confidence[]}      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Reasoning & Response Engine                        │  │
│  │                           (LLM-A)                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │  Context Assembly:                                               │ │  │
│  │  │  • 识别到的概念及其定义                                           │ │  │
│  │  │  • 相关文档块（基于概念检索）                                      │ │  │
│  │  │  • 用户原始问题                                                   │ │  │
│  │  │  • 推理指令                                                       │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                        │  │
│  │                              ▼                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │  Multi-Step Reasoning:                                           │ │  │
│  │  │  1. 概念关联分析：识别的概念之间有何联系？                         │ │  │
│  │  │  2. 证据整合：哪些文档支持这些判断？                               │ │  │
│  │  │  3. 推理链构建：如何从概念到答案？                                 │ │  │
│  │  │  4. 答案生成：综合所有信息回答用户                                 │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────┐                                                          │
│  │   Output     │                                                          │
│  │              │                                                          │
│  │ • 结构化答案 │                                                          │
│  │ • 概念标注   │                                                          │
│  │ • 证据链接   │                                                          │
│  │ • 置信度     │                                                          │
│  └──────────────┘                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件说明

#### 2.2.1 Knowledge Base (KB)

知识库包含两个核心存储：

**Concept Store（概念库）**

```yaml
concept:
  id: "LICR"
  name: "LLM-based Implicit Concept Recognition"
  
  # 自然语言定义（核心）
  definition: |
    LICR是一种利用大语言模型识别文本中隐性概念的技术。
    其特征包括：
    1. 识别对象是隐含但未明示的方法论、原理或模式
    2. 依赖语义理解而非关键词匹配
    3. 识别的是抽象模式而非具体实体
  
  # 原子分解（用于增强识别）
  atoms:
    - name: "Implicit"
      description: "概念存在但未被显式命名"
      synonyms: ["隐含", "潜在", "未明示", "暗含"]
    - name: "Pattern"
      description: "方法论、原理或可复用的模式"
      synonyms: ["方法论", "原理", "模式", "范式"]
    - name: "Recognition"
      description: "识别、发现、提取的过程"
      synonyms: ["识别", "发现", "检测", "提取"]
  
  # 正例示例
  positive_examples:
    - text: "这篇论文虽然没有提到反证法，但其证明过程明显采用了假设-矛盾-否定的结构"
      explanation: "识别了未明示的证明方法（反证法）"
    - text: "代码中存在一种未文档化的设计模式，通过分析调用关系才能发现"
      explanation: "识别了隐含的设计模式"
  
  # 负例示例（边界区分）
  negative_examples:
    - text: "论文明确使用了反证法来证明这个定理"
      explanation: "概念被显式提及，不是隐性识别"
    - text: "通过关键词搜索找到了相关文档"
      explanation: "关键词匹配，不是语义识别"
```

**Document Store（文档库）**

```yaml
document:
  id: "doc_001"
  source: "research_paper.pdf"
  
  chunks:
    - id: "chunk_001"
      text: "..."
      
      # LICR标注结果
      concept_annotations:
        - concept_id: "LICR"
          confidence: 0.92
          evidence: "文本描述了识别隐含模式的过程"
          annotated_by: "system"
          timestamp: "2026-01-12T10:00:00Z"
```

#### 2.2.2 Embedding Layer（嵌入预筛选层）

在调用LLM判断之前，使用向量检索进行快速预筛选：

```python
class EmbeddingFilter:
    def __init__(self, embedding_model="text-embedding-3-small"):
        self.model = embedding_model
        self.concept_embeddings = {}  # 预计算的概念定义embedding
    
    def prefilter(self, text: str, top_k: int = 10) -> List[str]:
        """
        快速筛选出最相关的候选概念
        避免对所有概念都调用LLM判断
        """
        text_embedding = self.encode(text)
        
        similarities = {}
        for concept_id, concept_emb in self.concept_embeddings.items():
            sim = cosine_similarity(text_embedding, concept_emb)
            similarities[concept_id] = sim
        
        # 返回相似度最高的top_k个概念
        return sorted(similarities, key=similarities.get, reverse=True)[:top_k]
```

#### 2.2.3 LICR Recognition Engine（识别引擎）

**LLM-C: Fast Judge（快速判断模型）**

```python
class FastJudge:
    """
    使用小模型进行快速的概念匹配判断
    核心思想：将识别问题转化为ICL推理问题
    """
    
    def __init__(self, model="Qwen2.5-1.5B-Instruct"):
        self.model = model
    
    def judge(self, concept_definition: str, text: str) -> JudgeResult:
        prompt = f"""任务：判断以下文本是否体现了给定概念。

概念定义：
{concept_definition}

待判断文本：
{text}

请分析：
1. 文本中是否存在与概念定义相符的特征？
2. 如果存在，具体体现在哪些方面？
3. 置信度评估（0-1）

输出格式：
{{
  "matches": true/false,
  "confidence": 0.0-1.0,
  "evidence": "具体证据说明",
  "reasoning": "判断理由"
}}"""
        
        return self.model.generate(prompt)
```

**LLM-B: Deep Judge（深度判断模型）**

```python
class DeepJudge:
    """
    对边界案例进行深度分析
    使用原子级别匹配和多步推理
    """
    
    def __init__(self, model="Qwen2.5-7B-Instruct"):
        self.model = model
    
    def deep_analyze(self, concept: Concept, text: str) -> DeepAnalysisResult:
        # 构建原子级别的分析prompt
        atoms_description = "\n".join([
            f"- {atom.name}: {atom.description}"
            for atom in concept.atoms
        ])
        
        prompt = f"""任务：对文本进行深度概念分析。

目标概念：{concept.name}
概念定义：{concept.definition}

概念的构成原子：
{atoms_description}

待分析文本：
{text}

请进行原子级别分析：
1. 逐一检查每个原子是否在文本中有体现
2. 对于每个匹配的原子，指出具体的文本证据
3. 综合判断：所有必要原子是否都存在？
4. 给出最终结论和置信度

输出格式：
{{
  "atom_analysis": [
    {{"atom": "...", "present": true/false, "evidence": "..."}}
  ],
  "overall_match": true/false,
  "confidence": 0.0-1.0,
  "detailed_reasoning": "..."
}}"""
        
        return self.model.generate(prompt)
```

#### 2.2.4 Reasoning & Response Engine（推理响应引擎）

```python
class ReasoningEngine:
    """
    基于识别结果进行多步推理，生成最终答案
    """
    
    def __init__(self, model="Claude-3.5-Sonnet"):
        self.model = model
    
    def reason_and_respond(
        self,
        user_query: str,
        recognition_results: List[RecognitionResult],
        retrieved_documents: List[Document]
    ) -> Response:
        
        # 组装上下文
        context = self._assemble_context(
            recognition_results,
            retrieved_documents
        )
        
        prompt = f"""你是一个基于知识库的智能助手。

## 已识别的概念
{self._format_concepts(recognition_results)}

## 相关文档
{self._format_documents(retrieved_documents)}

## 用户问题
{user_query}

## 任务
请基于以上信息回答用户问题。要求：
1. 优先使用识别到的概念来组织答案
2. 引用具体的文档证据
3. 如果信息不足，明确说明
4. 提供置信度评估

请开始回答："""
        
        return self.model.generate(prompt)
```

---

## 三、核心工作流程

### 3.1 新文本入库流程

```
新文本输入
    │
    ▼
┌─────────────────┐
│  文本预处理     │
│  • 分块         │
│  • 清洗         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding预筛选 │
│  获取Top-K候选   │
│  概念           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  LLM-C快速判断   │─────►│  高置信度结果    │
│  (并行处理K个    │      │  直接标注        │
│   候选概念)      │      └─────────────────┘
└────────┬────────┘
         │ 边界案例
         ▼
┌─────────────────┐      ┌─────────────────┐
│  LLM-B深度分析   │─────►│  最终标注结果    │
│  (原子级匹配)    │      └─────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  存入文档库      │
│  带概念标注      │
└─────────────────┘
```

### 3.2 用户查询响应流程

```
用户提问
    │
    ▼
┌─────────────────┐
│  问题理解       │
│  • 意图识别     │
│  • 关键概念提取 │
└────────┬────────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│  概念匹配检索    │          │  语义相似检索    │
│  基于LICR标注   │          │  传统RAG方式     │
└────────┬────────┘          └────────┬────────┘
         │                            │
         └──────────────┬─────────────┘
                        ▼
              ┌─────────────────┐
              │  结果融合排序    │
              │  去重+相关性排序 │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  LLM-A推理生成   │
              │  • 概念关联分析  │
              │  • 证据整合     │
              │  • 答案生成     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  结构化响应     │
              │  • 答案         │
              │  • 概念标注     │
              │  • 证据链接     │
              └─────────────────┘
```

---

## 四、模型选型与配置

### 4.1 模型角色分配

| 角色 | 推荐模型 | 参数量 | 部署方式 | 职责 |
|------|---------|--------|---------|------|
| LLM-A | Claude-3.5-Sonnet / GPT-4o | - | API调用 | 复杂推理、答案生成 |
| LLM-B | Qwen2.5-7B-Instruct | 7B | 本地GPU | 深度概念分析 |
| LLM-C | Qwen2.5-1.5B-Instruct / Phi-3-mini | 1.5B-3.8B | 本地CPU/GPU | 快速概念判断 |
| Embedding | text-embedding-3-small | - | API调用 | 向量检索预筛选 |

### 4.2 LLM-C选型对比

| 模型 | 参数量 | NLI性能 | CPU推理 | GPU推理 | 推荐场景 |
|------|--------|---------|---------|---------|---------|
| Qwen2.5-0.5B | 0.5B | ~78% | 极快 | - | 大规模预筛选 |
| Qwen2.5-1.5B | 1.5B | ~85% | 快 | 极快 | 平衡方案(推荐) |
| Phi-3-mini | 3.8B | ~88% | 较慢 | 快 | 高准确率需求 |
| DeBERTa-v3-base | 86M | ~90% | 极快 | - | NLI专用 |

### 4.3 级联策略配置

```yaml
cascade_config:
  # 预筛选阶段
  embedding_filter:
    top_k: 15  # 每个文本块筛选15个候选概念
    threshold: 0.3  # 最低相似度阈值
  
  # 快速判断阶段
  fast_judge:
    model: "Qwen2.5-1.5B-Instruct"
    high_confidence_threshold: 0.85  # 高于此值直接采纳
    low_confidence_threshold: 0.35   # 低于此值直接拒绝
    batch_size: 8  # 并行判断数量
  
  # 深度分析阶段
  deep_judge:
    model: "Qwen2.5-7B-Instruct"
    trigger_range: [0.35, 0.85]  # 触发深度分析的置信度区间
    max_concepts_per_text: 5  # 每个文本最多深度分析5个概念
```

---

## 五、概念定义规范

### 5.1 定义结构要求

为了最大化ICL识别效果，概念定义需要遵循以下结构：

```yaml
concept_definition_template:
  # 必需字段
  required:
    - id: 唯一标识符
    - name: 概念名称
    - definition: 自然语言定义（1-3段）
    
  # 推荐字段（提升识别准确率）
  recommended:
    - atoms: 原子分解（3-5个核心特征）
    - positive_examples: 正例（2-3个）
    - negative_examples: 负例/边界案例（2-3个）
    
  # 可选字段
  optional:
    - synonyms: 同义词/别名
    - related_concepts: 相关概念
    - domain: 所属领域
```

### 5.2 定义编写指南

**原则1：使用公共概念的组合**

```yaml
# ❌ 不好的定义（使用私有术语）
definition: "LICR是一种NER的扩展技术"

# ✅ 好的定义（分解为公共概念）
definition: |
  LICR是一种概念识别技术，其特征包括：
  1. 识别对象是"隐含的"（存在但未被明确提及）
  2. 识别内容是"方法论或模式"（而非实体名称）
  3. 识别方式是"语义理解"（而非关键词匹配）
```

**原则2：包含边界区分**

```yaml
# 明确说明"什么不是"
negative_examples:
  - text: "论文中明确提到使用了反证法"
    why_not: "概念被显式命名，不是隐性识别"
  - text: "通过搜索'反证法'关键词找到相关段落"
    why_not: "关键词匹配，不是语义识别"
```

**原则3：多变体表述**

```yaml
# 在示例中使用多种表述方式，增强泛化
positive_examples:
  - "文本中存在一种未命名的推理模式..."
  - "虽然没有提及术语，但逻辑结构显示..."
  - "这种隐藏的方法论可以通过分析得出..."
```

---

## 六、性能优化策略

### 6.1 批处理优化

```python
class BatchProcessor:
    """
    批量处理多个文本块的概念识别
    """
    
    async def process_batch(
        self,
        texts: List[str],
        concepts: List[Concept]
    ) -> List[RecognitionResult]:
        
        # 1. 并行Embedding计算
        embeddings = await self.embedding_model.encode_batch(texts)
        
        # 2. 批量预筛选
        candidate_pairs = []
        for i, text in enumerate(texts):
            top_concepts = self.filter_by_embedding(
                embeddings[i], concepts, top_k=10
            )
            for concept in top_concepts:
                candidate_pairs.append((text, concept))
        
        # 3. LLM-C批量判断（利用batch推理）
        fast_results = await self.fast_judge.judge_batch(candidate_pairs)
        
        # 4. 筛选需要深度分析的
        deep_candidates = [
            (text, concept, result)
            for (text, concept), result in zip(candidate_pairs, fast_results)
            if 0.35 < result.confidence < 0.85
        ]
        
        # 5. LLM-B深度分析（数量较少，可串行）
        deep_results = await self.deep_judge.analyze_batch(deep_candidates)
        
        return self.merge_results(fast_results, deep_results)
```

### 6.2 缓存策略

```yaml
cache_config:
  # 概念定义缓存（不常变化）
  concept_cache:
    type: "in-memory"
    ttl: 3600  # 1小时
    
  # Embedding缓存
  embedding_cache:
    type: "redis"
    ttl: 86400  # 24小时
    key_pattern: "emb:{text_hash}"
    
  # 判断结果缓存（可选，用于重复文本）
  judgment_cache:
    type: "redis"
    ttl: 604800  # 7天
    key_pattern: "judge:{text_hash}:{concept_id}"
```

### 6.3 预期性能指标

| 场景 | 文本块数量 | 概念数量 | 预期耗时 | 主要瓶颈 |
|------|-----------|---------|---------|---------|
| 单文本入库 | 1 | 100 | <2s | LLM-C推理 |
| 批量入库 | 100 | 100 | <30s | 并行LLM-C |
| 用户查询 | 1 | 100 | <3s | LLM-A生成 |
| 大规模索引 | 10000 | 100 | <10min | Embedding+批处理 |

---

## 七、与现有系统的集成

### 7.1 作为独立服务

```
┌─────────────────────────────────────────────────┐
│                 External Systems                 │
│  (应用程序、Web服务、数据管道)                    │
└─────────────────────┬───────────────────────────┘
                      │ REST API / gRPC
                      ▼
┌─────────────────────────────────────────────────┐
│                  ALL Service                     │
│  ┌───────────────────────────────────────────┐  │
│  │  API Layer                                 │  │
│  │  • POST /index     (索引新文本)            │  │
│  │  • POST /query     (查询)                  │  │
│  │  • GET  /concepts  (概念管理)              │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 7.2 API设计

```yaml
openapi: 3.0.0
paths:
  /index:
    post:
      summary: 索引新文本
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                text: 
                  type: string
                  description: 待索引的文本内容
                metadata:
                  type: object
                  description: 可选的元数据
      responses:
        200:
          content:
            application/json:
              schema:
                type: object
                properties:
                  document_id: string
                  concepts_detected:
                    type: array
                    items:
                      type: object
                      properties:
                        concept_id: string
                        confidence: number
                        evidence: string

  /query:
    post:
      summary: 基于知识库回答问题
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                question:
                  type: string
                context:
                  type: string
                  description: 可选的额外上下文
      responses:
        200:
          content:
            application/json:
              schema:
                type: object
                properties:
                  answer: string
                  concepts_used:
                    type: array
                  sources:
                    type: array
                  confidence: number
```

---

## 八、关键技术决策说明

### 8.1 为什么用ICL而非微调？

| 考量 | 微调方案 | ICL方案（本架构） |
|------|---------|-----------------|
| 跨表述泛化 | ❌ Reversal Curse问题 | ✅ 直接对比推理 |
| 概念更新 | 需要重新微调 | 修改定义文本即可 |
| 多概念支持 | 每个概念需训练数据 | 只需添加定义 |
| 部署成本 | 需要训练和存储多个LoRA | 无需训练 |
| 可解释性 | 黑盒判断 | 可输出推理过程 |

### 8.2 为什么采用级联架构？

```
如果所有判断都用大模型：
  100概念 × 1文本 × 2秒/次 = 200秒 ❌

级联架构：
  Embedding预筛选: 100→15概念, 0.1秒
  LLM-C快速判断: 15概念, 1.5秒
  LLM-B深度分析: ~3概念, 3秒
  总计: ~5秒 ✅
```

### 8.3 为什么需要原子分解？

基于EMNLP 2024的发现：单一表述训练无法实现跨表述泛化。

原子分解的作用：
1. 每个原子是模型预训练时见过的公共概念
2. 原子级别的识别泛化性更强
3. 组合规则相对简单，模型易于学习

---

## 九、实施路线图

### Phase 1: 核心验证（2周）

- [ ] 实现LLM-C快速判断模块
- [ ] 构建10个示例概念定义
- [ ] 验证ICL识别准确率
- [ ] 建立评估基准

### Phase 2: 完整流程（3周）

- [ ] 实现Embedding预筛选
- [ ] 实现LLM-B深度分析
- [ ] 实现级联流程
- [ ] 性能优化

### Phase 3: 系统集成（2周）

- [ ] API服务封装
- [ ] 知识库管理界面
- [ ] 监控和日志
- [ ] 文档完善

### Phase 4: 生产部署（2周）

- [ ] 压力测试
- [ ] 安全审计
- [ ] 部署上线
- [ ] 运维支持

---

## 十、附录

### A. 技术调研参考

本架构设计基于以下关键研究发现：

1. **Reversal Curse** (Berglund et al., ICLR 2024)
   - LLM微调后无法实现"A is B" → "B is A"的泛化
   - 即使是GPT-4也存在此问题

2. **Fine-Tuning or Retrieval?** (Ovadia et al., EMNLP 2024)
   - RAG在新知识场景下显著优于微调
   - 需要多变体训练才能部分缓解泛化问题

3. **In-Context Learning能力**
   - LLM在上下文中可以完成微调无法完成的推理
   - 这是ALL架构的核心理论基础

### B. 概念定义示例库

完整的概念定义示例见：`/examples/concept_definitions/`

### C. 评估指标定义

| 指标 | 定义 | 目标值 |
|------|------|--------|
| 准确率 | 正确识别 / 总识别 | >85% |
| 召回率 | 正确识别 / 应识别 | >80% |
| F1 Score | 准确率和召回率的调和平均 | >82% |
| 平均响应时间 | 从请求到响应的时间 | <3秒 |

---

**文档结束**

*ALL: Augmented LLM with LICR - 让知识库真正理解你的领域*
