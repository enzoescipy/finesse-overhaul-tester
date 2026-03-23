# MTEB Model Selection Log for FINESSE Benchmark

**Date of Operation:** 2026-03-24

## Phase 1: Data Acquisition

- The complete Massive Text Embedding Benchmark (MTEB) leaderboard dataset was downloaded at `2026-03-24 00:42 UTC+9`.
- The resulting dataset is stored as `tmpukyk5dfv.csv`.

## Phase 2: Automated Filtering

**Objective:** To build a representative and unbiased model pool for the FINESSE benchmark evaluations.

### Filtering Criteria

The following automated filters were applied to the MTEB leaderboard:

1.  **Context Window Threshold:** `Context Window >= 8190` tokens.
    -   *Rationale: Essential for meaningful long-context analysis.*
2.  **Parameter Ceiling:** `Parameters < 8.0B`.
    -   *Rationale: Focus on efficient, widely accessible models.*
3.  **Open Source Mandate:** The model must be open source.
    -   *Rationale: Ensures reproducibility and transparency, which are core to our research values.*

### Filtering and Sampling Procedure

- The filtering and shuffling logic is implemented in the `filter_mteb.py` script.
- The script was executed with the command: `python filter_mteb.py --num-samples 10000 --seed 42`.
- The initial filtered list is saved as `filtered_models.csv`.

## Phase 3: Manual Curation and Verification

**Objective:** To manually verify model compatibility and apply selection criteria not suitable for automation.

### Manual Filtering Criteria

1.  **Loading Compatibility:** Must be loadable via the standard Hugging Face `transformers` library without custom code.
2.  **Language Support:** Must primarily support English, as our evaluation datasets are in English.
3.  **Architectural Scope:** Must produce a single-vector representation (i.e., dense retrieval models). Late-interaction models (e.g., ColBERT) and rerankers were excluded.
4.  **License Clarity:** Must have a clear and permissible license for academic and research use.

### Manually Excluded Models

The following models were excluded during manual curation for the reasons specified:

-   `McGill-NLP/LLM2Vec models`: Ambiguous inference procedure makes reproducible evaluation difficult.
-   `GritLM/GritLM-7B`: Requires custom model loading procedures outside the standard `transformers` pipeline.
-   `intfloat/e5-mistral-7b-instruct`: While listing a 32,768 token limit, the official model card warns: `"Using this model for inputs longer than 4096 tokens is not recommended."` This makes it unsuitable for our long-context focus.
-   `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`: Primarily a Vision-Language model, which is outside the scope of this text-based research.
-   `Alibaba-NLP/gme-Qwen2-VL-7B-Instruct`: Same reason as above.

### Final Result

-   **Final Model Count:** 75
-   The final, curated list of models is saved as `manually_selected_model.csv`.

### Evaluation Fixage

- All model would be evaluated by the `transformer` python library inference.
- Pooling method must be one of : `mean pooling`, `cls pooling(first token selection)`, `last token pooling`.
- If the model recommend to use the custom inference pipeline, base mode will be chosen.