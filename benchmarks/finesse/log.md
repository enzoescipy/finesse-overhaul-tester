## Prepare of the Experiments

We wrote the two scripts : `preset/generate_eval_finesse_configs.py` and `preset/evaluate_finesse_automate.py`.
- `preset/generate_eval_finesse_configs.py` first generate the `srs.yaml` & `rss.yaml`.
- `preset/evaluate_finesse_automate.py` evaluate the model wrote on the `srs.yaml` & `rss.yaml`, half-automatically.

All task evaluated on Colab G4 instance. (NVIDIA RTX PRO 6000 Blackwell Server Edition). 

batch size unified size=32 for srs evaluation. rss evaluation is single-batch support only, as the `finesse-benchmark` package.

We installed following packages.

- finesse_benchmark==0.18.8

We evaluated the models listed on the `model-selection\model-verify-lemb-log\investigation-verified.csv`. see the `model-selection\model-verify-lemb-log\log.md` and the `model-selection\readme.md` for detail.

## Result log of the Experiments

### 2026-03-29 : Experiment Finished

All model processed well on the experiments. Nothing special to record here. 

## Conclusion

-   **Final Model Count:** 34
-   Final FINESSE evaluation results are saved as,
    - `model_eval/model_eval_split_0.zip`
    - `model_eval/model_eval_split_1.zip`
    - `model_eval/model_eval_split_2.zip`
    - `model_eval/model_eval_split_3.zip`