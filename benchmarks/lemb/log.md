
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
