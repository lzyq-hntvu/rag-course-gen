# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based vocational course generation system (GraphRAG) that combines vector search with knowledge graphs. The system connects enterprise job descriptions (JD) to national vocational teaching standards, enabling automated course content generation for cybersecurity education.

**Core Architecture**: Two-phase hybrid retrieval
- Phase 1: Traditional vector RAG (ChromaDB + sentence-transformers)
- Phase 2: GraphRAG enhancement (NetworkX knowledge graph)

## Development Setup

```bash
# One-click environment setup
./setup.sh

# Manual activation
source .venv/bin/activate
# or
./activate.sh
```

**Required Configuration** (`.env`):
```bash
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4-plus
```

**Important**: This project uses ZhipuAI GLM-4 API (OpenAI-compatible). The embedding model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual, 384-dim). In China, HuggingFace mirrors are configured: `https://hf-mirror.com`.

## Common Commands

```bash
# Vector RAG pipeline
python src/ingest.py                 # Ingest documents from docs/ to ChromaDB
python src/test_retrieve.py          # Interactive retrieval testing

# GraphRAG pipeline
python src/graph/build_batch.py      # Build graph from JD data (main entry point)
python src/graph/visualize.py        # Visualize graph with matplotlib
python src/graph/validate.py         # 4-dimensional quality validation
```

**Note**: `build_batch.py` orchestrates the entire two-phase graph construction pipeline. Processing 50 JDs takes 10-15 minutes (~15-30s per JD with LLM calls).

## Architecture

### Two-Phase Graph Construction Pattern

The system uses a **skeleton-constrained extraction** approach:

```
Phase 1: Extract national standard tasks (skeleton framework)
    ↓
Phase 2: Extract JD entities constrained by skeleton tasks
    ↓
Build NetworkX graph with typed nodes/edges
```

This ensures JD entities map to standardized national framework tasks (see `src/graph/extractor.py:extract()` with `candidate_tasks` parameter).

### Entity Schema (4 types, 3 relations)

**Nodes** (`src/graph/schema.py`):
- `StandardTask`: Typical work tasks from national standards (course skeleton)
- `RealProject`: Real project carriers from JDs (teaching context), with `experience_level` field
- `ActionSkill`: Bridging skills connecting JD to standards, with `complexity` field
- `Tool`: Tools/equipment used by skills

**Edges**:
- `composed_of`: RealProject → ActionSkill
- `maps_to`: ActionSkill → StandardTask (core reasoning edge)
- `operates`: ActionSkill → Tool

**Layering System** (key for vocational differentiation):
- `ExperienceLevel`: entry → junior → mid_level → senior → expert
- `SkillComplexity`: low → medium → high

### Data Flow

```
data/raw/job_spider/ (CSV files)
    ↓ src/graph/loader.py (JobDataLoader)
JobRecord objects
    ↓ src/graph/extractor.py (EntityExtractor)
ExtractedEntities (with few-shot prompting)
    ↓ entities_to_schema()
GraphNode/GraphEdge objects
    ↓ src/graph/graph_builder.py (VocationalGraphBuilder)
NetworkX MultiDiGraph → data/output/graph.pkl
    ↓ src/graph/validate.py / visualize.py
Validation reports + visualizations
```

### Module Interconnections

- `config.py`: Centralized paths, chunk sizes, model names
- `graph/schema.py`: Pydantic models for all entities/relations
- `graph/extractor.py`: LLM-based extraction with constraint support
- `graph/loader.py`: CSV JD data loading with domain filtering
- `graph/graph_builder.py`: NetworkX wrapper with shortest path, PageRank
- `graph/build_batch.py`: Batch orchestration with filtering (by complexity/experience)
- `graph/validate.py`: 4-dimensional validation (completeness, connectivity, layering, reasoning)
- `graph/visualize.py`: Matplotlib visualization with color-coded entity types

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/config.py` | Centralized configuration (paths, models, chunking) |
| `src/ingest.py` | Document ingestion to ChromaDB vector store |
| `src/test_retrieve.py` | Interactive retrieval testing (batch/interactive modes) |
| `src/graph/schema.py` | **Core**: Pydantic models for 4 entity types, 3 relations, 2 layering enums |
| `src/graph/extractor.py` | **Core**: LLM entity extractor with few-shot prompting |
| `src/graph/loader.py` | CSV JD loader with domain keyword filtering |
| `src/graph/graph_builder.py` | NetworkX graph builder (nodes, edges, queries) |
| `src/graph/build_batch.py` | **Main entry**: Orchestrates two-phase extraction |
| `src/graph/validate.py` | Quality assurance (4-dimensional checks) |
| `src/graph/visualize.py` | Graph visualization with filtering support |

## Testing Strategy

- **Mock data**: `data/mock/sample_jds.json` (5 JDs), `data/mock/sample_standard.txt`
- **Real data**: `data/raw/job_spider/` (19,000+ files from job spider)
- **Validation**: Always run `validate.py` before deployment to check:
  1. Completeness (all nodes reachable)
  2. Connectivity (no isolated components)
  3. Layering (experience/complexity distribution)
  4. Reasoning paths (shortest path analysis)

## Output Artifacts

Located in `data/output/`:
- `graph.pkl` - Serialized NetworkX graph
- `graph_full.png` - Full visualization
- `graph_complexity_*.png` - Filtered by complexity level
- `graph_experience_*.png` - Filtered by experience level
