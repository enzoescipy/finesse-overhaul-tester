## Prepare of the Experiments

We wrote the two scripts : `preset/generate_eval_sffd_configs.py` and `preset/evaluate_sffd_automate.py`.
- `preset/generate_eval_sffd_configs.py` first generate the `config.json`.
- `preset/evaluate_sffd_automate.py` evaluate the model wrote on the `config.json`, half-automatically.

All task evaluated on Colab G4 instance. (NVIDIA RTX PRO 6000 Blackwell Server Edition). 

batch size vary for the architecture.

- Most of the transformer embedder : unified for single batch(=1).
- Encoder + Synthesizer set : Encoder's batch is unlimited. Synthesizer's batch is 1.

For current stage, All following selected model is the 'Most of the transformer embedder' case. Therefore, batch size is unified to single batch(=1).

We installed following packages.

- faiss-cpu==1.13.2
- mteb==2.10.14

We evaluated the models listed on the `model-selection\model-verify-lemb-log\investigation-verified.csv`. see the `model-selection\model-verify-lemb-log\log.md` and the `model-selection\readme.md` for detail.

## Result log of the Experiments

### 2026-03-26 : Experiment Finished

All model processed well on the experiments. Nothing special to record here. 

## Conclusion

-   **Final Model Count:** 34
-   Final SFfD evaluation results are saved as,
    - `model_eval/model_eval_split_0.zip`
    - `model_eval/model_eval_split_1.zip`


### 2026-04-04 : Additional Experiments Added

We evaluated additional method.

- `model_eval/model_eval_baseline.zip` : additional baseline model, which is the "chunk-embed-then-average" context extending strategy applied for the `intfloat/multilingual-e5-base` model.

no model list changed. additional baseline model should NOT be treated as the single transformer model evaluation result.

**evaluate_sffd_automate.py was modified for the baseline model evaluation.**