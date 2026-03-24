# Protocol Amendment Log (2026-03-24)

## 1. Context and Purpose

This document serves as an official amendment to the model evaluation protocol originally established on March 24, 2026, as detailed in the preceding log files.

Following the initial execution of the benchmark procedure, a critical parameter inconsistency was discovered that impacts the validity and reproducibility of the results for certain models on long-context tasks. This amendment details the nature of the issue, the corrective actions taken, and the revised methodology that will be applied uniformly to all subsequent experiments.

The primary objective of this action is to uphold the highest standards of scientific integrity and to ensure that the final benchmark results are robust, reliable, and founded on a sound experimental setup.

## 2. Discovery and Correction

We discovered that although [bge-m3](https://huggingface.co/BAAI/bge-m3) official huggingface repository's `config.json` reports **8194** `max_positional_embeddings` value(based on hash b28ce2a6fcc9c75ef1c0619575d0ec19af760082 commit), but in the actual testing environment, the model cannot inference and CUDA kernal crash happened. 

We then realized the official huggingface repository's `README.md` specified the `Sequence Length` value by **8192**, and it was the actual max_ctx of the model both the official announcement, and our test setup.

For this reason, We add the following rule for the model selection.

5. **All model will be reguarded as max_ctx=8190. If model fails, We will let them not tested further.** `Context Window >= 8190` tokens was our initial filtering logic for the investigation(see the`model-selection-log/log.md`). The `### Further max_ctx Investigation` section makes this decision clear, because there are too many model we cannot find the exact max_ctx the author of the model suggest, best in our knowledge. 

### Further max_ctx Investigation

We will highlight if the max_ctx is different from the original MTEB datasheet, like 514.0 -> **512**. Also we will highlight if we cannot find the information of the model's max_ctx, best in our knowledge, like **Unknown, but xxx source specify the 512.**. If the source of the max_ctx is different from the huggingface repository, we will also specify them.

Model | Max Tokens(original) | Max Tokens(official)
|------|----------------|------------------------|
[gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) | 32768.0 | **32000**
[F2LLM-v2-8B](https://huggingface.co/codefuse-ai/F2LLM-v2-8B) | 40960.0 | **Unknown**
[Zeta-Alpha-E5-Mistral](https://huggingface.co/zeta-alpha-ai/Zeta-Alpha-E5-Mistral) | 32768.0 | **not specified, but `How to Run` section specify 4096(on huggingface repository). [on blog post](https://www.zeta-alpha.com/post/fine-tuning-an-llm-for-state-of-the-art-retrieval-zeta-alpha-s-top-10-submission-to-the-the-mteb-be) -> trained on doc/query = 512/192.**
[snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) | 8192.0 | 8192
[jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) | 8192.0 | 8192 
[Linq-Embed-Mistral](https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral) | 32768.0 | **not specified, but `How to use` section specify 4096(on huggingface repository)** 
[llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b) | 32768.0 | 32768 
[bge-m3-unsupervised](https://huggingface.co/BAAI/bge-m3-unsupervised) | 8192.0 | 8192
[SFR-Embedding-2_R](https://huggingface.co/Salesforce/SFR-Embedding-2_R) | 32768.0 | **not specified, but `How to Run` section specify 4096(on huggingface repository)**
[gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 8192.0 | 8192
[F2LLM-v2-0.6B](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B) | 40960.0 | **Unknown**
[speed-embedding-7b-instruct](https://huggingface.co/Haon-Chen/speed-embedding-7b-instruct) | 32768.0 | **4096. `Using this model for inputs longer than 4096 tokens is not recommended` by the author, in the huggingface repository.**
[gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | 8192.0 | 8192 
[jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 32768.0 | 32768
[jina-embeddings-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en) | 8192.0 | 8192 
[jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) | 8192.0 | 8192
[inf-retriever-v1-1.5b](https://huggingface.co/infly/inf-retriever-v1-1.5b) | 32768.0 | 32768
[gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) | 8192.0 | 8192
[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 8192 | (not changed)
[F2LLM-v2-4B](https://huggingface.co/codefuse-ai/F2LLM-v2-4B) | 40960.0 | **Unknown**
[F2LLM-4B](https://huggingface.co/codefuse-ai/F2LLM-4B) | 8192.0 | **Unknown**
[pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) | 32768.0 | **32000, we interpret `32k` as equal to 32000.**
[Qodo-Embed-1-7B](https://huggingface.co/Qodo/Qodo-Embed-1-7B) | 32768.0 | **32000, we interpret `32k` as equal to 32000.**
[snowflake-arctic-embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) | 8192.0 | 8192
[F2LLM-v2-1.7B](https://huggingface.co/codefuse-ai/F2LLM-v2-1.7B) | 40960.0 | **Unknown**
[NV-Embed-v1](https://huggingface.co/nvidia/NV-Embed-v1) | 32768.0 | **not specified, but `Usage` section specify 4096(on huggingface repository)**
[F2LLM-v2-330M](https://huggingface.co/codefuse-ai/F2LLM-v2-330M) | 40960.0 | **Unknown**
[bge-m3](https://huggingface.co/BAAI/bge-m3) | 8194.0 | **8192**
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 32768.0 | **32000, we interpret `32k` as equal to 32000.**
[QZhou-Embedding](https://huggingface.co/Kingsoft-LLM/QZhou-Embedding) | 8192.0 | **8000, we interpret `8k` as equal to 8000. however, the `Usage` section shows the 8192 as example.**
[BOOM_4B_v1](https://huggingface.co/ICT-TIME-and-Querit/BOOM_4B_v1) | 32768.0 | **32000, we interpret `32k` as equal to 32000.**
[jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) | 8194.0 | **8192** 
[Octen-Embedding-8B](https://huggingface.co/bflhc/Octen-Embedding-8B) | 32768.0 | 32768
[gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) | 32768.0 | **32000**
[sarashina-embedding-v1-1b](https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b) | 8192.0 | **8192**
[jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small) | 32768.0 | 32768
[F2LLM-1.7B](https://huggingface.co/codefuse-ai/F2LLM-1.7B) | 8192.0 | **Unknown**
[LGAI-Embedding-Preview](https://huggingface.co/annamodels/LGAI-Embedding-Preview) | 32768.0 | **32000, we interpret `32k` as equal to 32000.**
[Octen-Embedding-4B](https://huggingface.co/bflhc/Octen-Embedding-4B) | 32768.0 | 32768
[SearchMap_Preview](https://huggingface.co/VPLabs/SearchMap_Preview) | 8192.0 | **Unknown. **
[granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) | 8192.0 | (not changed)
[Octen-Embedding-0.6B](https://huggingface.co/bflhc/Octen-Embedding-0.6B) | 32768.0 | (not changed)
[F2LLM-0.6B](https://huggingface.co/codefuse-ai/F2LLM-0.6B) | 8192.0 | **Unknown**
[granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) | 8192.0 | (not changed)
[Qodo-Embed-1-1.5B](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B) | 32768.0 | (not changed)
[F2LLM-v2-80M](https://huggingface.co/codefuse-ai/F2LLM-v2-80M) | 40960.0 | **Unknown**
[GeoEmbedding](https://huggingface.co/GeoGPT-Research-Project/GeoEmbedding) | 32768.0 | **Unknown. But author note that this model is fine-tuned on `NovaSearch/stella_en_400M_v5`, which is 512 max_ctx.**
[nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1) | 8192.0 | 8192
[F2LLM-v2-160M](https://huggingface.co/codefuse-ai/F2LLM-v2-160M) | 40960.0 | **Unknown**
[MoD-Embedding](https://huggingface.co/bflhc/MoD-Embedding) | 32768.0 |  32,768
[Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 32768.0 | **32000, we interpret `32k` as equal to 32000.**
[Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 32768.0 | **32000, we interpret `32k` as equal to 32000.**
[NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) | 32768.0 | **not specified, but `How to use` section specify 32768(on huggingface repository)**
[modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base) | 8192.0 | **Unknown**
[Solon-embeddings-mini-beta-1.1](https://huggingface.co/OrdalieTech/Solon-embeddings-mini-beta-1.1) | 8192.0 | **Unknown**
[gte-Qwen1.5-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct) | 32768.0 | **32000, we interpret `32k` as equal to 32000.**
[pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b) | 32768.0 | **32000, we interpret `32k` as equal to 32000.**
[SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral) | 32768.0 | **not specified, but `How to Run` section specify 4096(on huggingface repository). [on blog post](https://www.salesforce.com/blog/sfr-embedding/) -> trained on doc/query = 256/128.**
[inf-retriever-v1](https://huggingface.co/infly/inf-retriever-v1) | 32768.0 | 32768

## Conclusion

-   **Final Model Count:** 59 (not changed)
-   The revised list of models and the configuration is saved as `investigation-revised0.csv`.
