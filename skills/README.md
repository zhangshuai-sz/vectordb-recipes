
# Skills 知识库

本目录用于沉淀在使用 [vectordb-recipes](https://github.com/lancedb/vectordb-recipes) 各个 demo 过程中积累的实战经验、踩坑记录与最佳实践。

每个子目录对应一个具体的 demo / 技术方案，作为可复用的"skill 卡片"。

---

## 目录结构

```
skills/
├── README.md                          # 本文件：总索引
└── multimodal_clip_diffusiondb/       # CLIP + DiffusionDB 多模态图像检索
    └── README.md
```

---

## Skill 索引

| Skill | 核心模型 | 数据 | 形态 | 状态 |
| --- | --- | --- | --- | --- |
| [multimodal_clip_diffusiondb](./multimodal_clip_diffusiondb/README.md) | OpenAI CLIP (ViT-B/32) | DiffusionDB AI 生成图 | Gradio Web UI | ✅ 已沉淀 |
| _multimodal_jina_clipv2_ | Jina-CLIP v2（多语言） | 图文对 | Notebook | 🔲 待补充 |
| _multimodal_video_search_ | OpenAI CLIP | YouTube 8M 视频帧 | Gradio Web UI | 🔲 待补充 |
| _voyagexlancedb_ | Voyage AI（API） | Conceptual Captions | Notebook | 🔲 待补充 |
| _v-jepa-video-search_ | V-JEPA 2（自监督视频） | 单视频内帧 | Notebook | 🔲 待补充 |

---

## 共性技术栈

所有 demo 都基于：

- **向量库**：[LanceDB](https://lancedb.com/) — 列式存储 + 向量索引一体化
- **底层格式**：Lance（基于 Arrow 的列存格式，支持 zero-copy）
- **多模态 Embedding**：CLIP 系或新一代多模态模型
- **常见检索模式**：
  - 向量检索（vector search）
  - 全文检索（FTS / BM25）
  - 混合检索（hybrid）
  - SQL 过滤（DuckDB 直接读 lance 文件）

---

## 使用约定

每个 skill 子目录的 `README.md` 应至少包含以下章节：

1. **Demo 简介**：用途、效果截图
2. **目录结构**：源码文件作用
3. **核心流程**：从原始数据 → embedding → 入库 → 检索的完整链路
4. **关键代码片段**：带行号或函数名的核心实现
5. **常见踩坑**：版本兼容、API 变更、性能问题等
6. **运行指南**：环境、命令、端口、外网访问方式
7. **可优化方向**：模型升级、性能调优、效果改进的可能路径
