# MTEB Model Investigation Log for FINESSE Benchmark

**Date of Operation:** 2026-03-24

## field description

- Pooling Method : one of `last`, `mean`, `cls`.
- Is Instructed : `yes` or `no`. if `yes`, automatically query prefix will be `Instruct:{} \nQuery:`.
- Query Prefix & Document Prefix : special prefix that should be put on the very first of embeded text, as the constant.

## Additional Changes

- Removal of `yibinlei/LENS-d4000`, `yibinlei/LENS-d8000` : max pooling is prohibited.
- Removal of `BAAI/bge-en-icl` : this is few-shot in-context learning embedder. we cannot evaluate its performance fairly.
- Removal of `infgrad/jasper_en_vision_language_v1` : this is vision-language-embedder. we had to filter it before. 
- Removal of `infgrad/Jasper-Token-Compression-600M` : Sadly, we could not fully understood how this model is working. 
- Removal of `zeroentropy/zembed-1` : `transformer` library usage not specified.
- Usage of jina models family : all jina family model will be evaluated by proper pooling method, but WITHOUT the proper lora adapter and favored query & document prefix. Evaluation pipeline dose not support the custom lora adapter selection. Will use the base model and no prefix option.
- Removal of `nomic-ai/nomic-embed-text-v1-unsupervised`, `nomic-ai/nomic-embed-text-v1-ablated` : Author declared this is the checkpoint purpose repository for the further model.
- Usage of NV-Embed models : Author specified the pooling method to "Latent-Attention" pooling. We will interpret this to `last` pooling method.
- Usage of `annamodels/LGAI-Embedding-Preview` : No pooling method or instruction specfied. We will interpret as `last` pooling with `Instruct:{} \nQuery:` style embedder.
- Usage of `VPLabs/SearchMap_Preview` : No pooling method or instruction specfied. We will interpret as `cls` pooling without prefix.
- Removal of `fyaronskiy/english_code_retriever` : No Explicit licence specified. we had to filter it before.
- Removal of `codefuse-ai/C2LLM-0.5B`, `codefuse-ai/C2LLM-7B` : They suggest the novel pooling method, PMA (Pooling by Multi-head Attention), which cannot be evaluated for now.
- Removal of `NovaSearch/stella_en_1.5B_v5`, `NovaSearch/stella_en_400M_v5`: This model is known as max ctx of >8192, but we found the author note that : `Q: What is the sequence length of models? A: 512 is recommended, in our experiments, almost all models perform poorly on specialized long text retrieval datasets. Besides, the model is trained on datasets of 512 length. This may be an optimization term.`
- Removal of `Tarka-AIR/Tarka-Embedding-350M-V1` : which uses `lfm1.0` special licence, giving the sence of ambiguity.
- Removal of `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte` : special "Sparse Vector" embedding mechanism included. cannot evaluated for the standard `transformer` library use.

# Final Declare

-   **Final Model Count:** 60
-   The final, curated list of models is saved as `investigation.csv`.
