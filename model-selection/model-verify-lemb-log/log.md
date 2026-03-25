
## Prepare of the Experiments

We wrote the two scripts : `preset/generate_eval_lemb_configs.py` and `preset/evaluate_lemb_automate.py`.
- `preset/generate_eval_lemb_configs.py` first generate the `config.json`.
- `preset/evaluate_lemb_automate.py` evaluate the model wrote on the `config.json`, half-automatically.

All task evaluated on Colab G4 instance. (NVIDIA RTX PRO 6000 Blackwell Server Edition). batch size all unified of single batch(=1).

We installed following packages.

- faiss-cpu==1.13.2
- mteb==2.10.14

All code referenced from [LongEmbed](https://github.com/dwzhu-pku/LongEmbed).
Only minimum-code change was occoured from the original code, However, 

```python
    # WINDOW_LENGTH_LIST = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    # We stripped the model's max_ctx to 8190. See the main paper and appendix for the detailed reason for this setting.
    # This is the only difference from the original LEMB scoring.
    WINDOW_LENGTH_LIST = [256, 512, 1024, 2048, 4096, 8192]
```

This is intended. We select to cut-off the model's `max_ctx` to `8190` (see `model-selection\model-investigation-revised0-log\log.md`). It makes us difficult to evaluate the `16384`, `32768` section of `Needle` and `Passkey` Task. We decided to exclude the `16384`, `32768`length test.

## Result Log of the Experiments

### 2026-03-25 : partial LEMB evaluation failed

Some models failed to load due to dependency conflicts (e.g., No module named 'transformers.models.qwen2.tokenization_qwen2_fast'). This was likely caused by a mismatch between the Colab instance's transformers version and the version required by the model. These models were immediately excluded from the current run.

However, many models caused a fatal CUDA kernel crash, which prevented any traceback from being generated. These models were also marked as failed.

A critical issue observed was that a CUDA crash from one model could corrupt the runtime environment. Since the automation script cannot automatically reboot the Colab instance after such a crash, subsequent models in the queue could also fail, not due to their own faults, but due to the contaminated environment. This necessitates a re-evaluation of models that failed after an initial CUDA crash.

Due to the script's limitations and the aforementioned failures, some models appear to have been evaluated twice, while others were skipped entirely. While duplicate evaluations are not a concern, skipped models require re-evaluation.

The raw results from this initial, partially-failed run have been archived in `benchmarks/lemb/model_eval_0.zip`. We will now proceed to:

1. Isolate the models that failed due to potential chain-failure contamination or were skipped.
2. Re-evaluate these specific models in a clean runtime environment using the exact same procedure.

### 2026-03-25 : failed evaluation re-evaluated result

The re-evaluation process yielded critical clarifications. Several models from the 'unknown-failure' and 'not-tested' groups were successfully evaluated and moved to the 'test-succeed' category. Conversely, some 'unknown-failure' models consistently reproduced fatal errors even in a clean environment, allowing us to move them confidently to the 'known-failure' group with a confirmed failure status.

We grouped the past `benchmarks/lemb/model_eval_0.zip` result into four category.

-   **known-failure:** If the failure of the model's causes is well known. `error.log` inside of the each model's folder explains the failure reason. 
-   **unknown-failure:** If the failure of the model's causes is just noted as like *CUDA error: device-side assert triggered*. It is potentially failed model without explicit causes but also can be the side-effect we discussed earlier.
-   **not-tested:** If not touched model caused by automation script's fault.
-   **test-succeed:** If test result and its evidence output is well-created.

We tested only not-tested, unknown-failure model. For this time, if the model created any kinds of interruption, we reboot the colab instance.

result of sorted, organized, and re-tested changed version of `model_eval_0.zip` is archived in `benchmarks/lemb/model_eval_1.zip`.  

We now move on next stage, which would be re-location of the succed experiment model inside `unknown-failure` or `not-tested` to the `test-succeed` group, and if exists, explicit failure reason being yield from new model inside of `unknown-failure` or `not-tested` group, moving them into `known-failure` group.

### 2026-03-25 : final model selection verification

We finally conclude to use / not to use the model, as the following reason.

#### Excluded : Unknown CUDA exception

We now verify this two model, 

- [gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)
- [gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)

In our test environment, they produce serious unknown CUDA "*CUDA error: device-side assert triggered*" exception, which pollute the excution instance, making other potentially healthy model also trigger the "*CUDA error: device-side assert triggered*" exception. 

#### Excluded : Known exception, like Environment configuration mismatch

##### Error: No module named 'transformers.models.qwen2.tokenization_qwen2_fast'

- [gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)
- [gte-Qwen1.5-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct)
- [gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
- [inf-retriever-v1-1.5b](https://huggingface.co/infly/inf-retriever-v1-1.5b)
- [inf-retriever-v1](https://huggingface.co/infly/inf-retriever-v1)
- [Qodo-Embed-1-7B](https://huggingface.co/Qodo/Qodo-Embed-1-7B)
- [Qodo-Embed-1-1.5B](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B)

##### Error: `num_hidden_layers` (x) must be equal to the number of layer types (y)

- [F2LLM-v2-80M](https://huggingface.co/codefuse-ai/F2LLM-v2-80M)
- [F2LLM-v2-160M](https://huggingface.co/codefuse-ai/F2LLM-v2-160M)
- [F2LLM-v2-330M](https://huggingface.co/codefuse-ai/F2LLM-v2-330M)

##### Error: No module named 'transformers.onnx'

- [jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)
- [jina-embeddings-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en)

##### Error: please install xformers

- [snowflake-arctic-embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)
- [SearchMap_Preview](https://huggingface.co/VPLabs/SearchMap_Preview)

##### Error: list index out of range

- [pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b)
- [pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b)

##### Etc.

- [GeoEmbedding](https://huggingface.co/GeoGPT-Research-Project/GeoEmbedding) : *Error: 'MistralDualModel' object has no attribute '_update_causal_mask'*
- [jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) : *Error: '<' not supported between instances of 'str' and 'int'*
- [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) : *Error: cannot import name 'SlidingWindowCache' from 'transformers.cache_utils' (/usr/local/lib/python3.12/dist-packages/transformers/cache_utils.py)*
- [QZhou-Embedding](https://huggingface.co/Kingsoft-LLM/QZhou-Embedding) : *Error: 'Qwen2Config' object has no attribute 'rope_theta'*
- [NV-Embed-v1](https://huggingface.co/nvidia/NV-Embed-v1) : *Error: cannot import name 'MISTRAL_INPUTS_DOCSTRING' from 'transformers.models.mistral.modeling_mistral' (/usr/local/lib/python3.12/dist-packages/transformers/models/mistral/modeling_mistral.py)*
- [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) : *Error: 'NVEmbedModel' object has no attribute 'all_tied_weights_keys'*
- [Solon-embeddings-mini-beta-1.1](https://huggingface.co/OrdalieTech/Solon-embeddings-mini-beta-1.1) : *Error: 'default'*

#### Final Included List of Models (LEMB evaluation completed)

- [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)
- [LGAI-Embedding-Preview](https://huggingface.co/annamodels/LGAI-Embedding-Preview)
- [bge-m3](https://huggingface.co/BAAI/bge-m3)
- [bge-m3-unsupervised](https://huggingface.co/BAAI/bge-m3-unsupervised)
- [MoD-Embedding](https://huggingface.co/bflhc/MoD-Embedding)
- [Octen-Embedding-0.6B](https://huggingface.co/bflhc/Octen-Embedding-0.6B)
- [Octen-Embedding-4B](https://huggingface.co/bflhc/Octen-Embedding-4B)
- [Octen-Embedding-8B](https://huggingface.co/bflhc/Octen-Embedding-8B)
- [F2LLM-0.6B](https://huggingface.co/codefuse-ai/F2LLM-0.6B)
- [F2LLM-1.7B](https://huggingface.co/codefuse-ai/F2LLM-1.7B)
- [F2LLM-4B](https://huggingface.co/codefuse-ai/F2LLM-4B)
- [F2LLM-v2-0.6B](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B)
- [F2LLM-v2-1.7B](https://huggingface.co/codefuse-ai/F2LLM-v2-1.7B)
- [F2LLM-v2-4B](https://huggingface.co/codefuse-ai/F2LLM-v2-4B)
- [F2LLM-v2-8B](https://huggingface.co/codefuse-ai/F2LLM-v2-8B)
- [speed-embedding-7b-instruct](https://huggingface.co/Haon-Chen/speed-embedding-7b-instruct)
- [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2)
- [granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)
- [BOOM_4B_v1](https://huggingface.co/ICT-TIME-and-Querit/BOOM_4B_v1)
- [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)
- [jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)
- [Linq-Embed-Mistral](https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral)
- [modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base)
- [nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b)
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- [SFR-Embedding-2_R](https://huggingface.co/Salesforce/SFR-Embedding-2_R)
- [SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral)
- [sarashina-embedding-v1-1b](https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b)
- [snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)
- [Zeta-Alpha-E5-Mistral](https://huggingface.co/zeta-alpha-ai/Zeta-Alpha-E5-Mistral)

## Conclusion

-   **Final Model Count:** 34 (changed, reduced)
-   The revised list of models and the configuration is saved as `investigation-verified.csv`.
-   Final LEMB evaluation results (including excluded models) are saved as `model_eval_2.zip`.
-   This `log.md` and `investigation-verified.csv` is copied to the `model-selection/model-verify-lemb-log` folder.