import shutil
from pathlib import Path
import os

UNZIP_CHART = {
    "srs-16" : "benchmarks/finesse/additional_model_eval/cultura-en-l16-srs-weak.zip",
    "rss-wikipedia" : "benchmarks/finesse/additional_model_eval/wikipedia-en-l15-rss.zip",
    "finesse-baseline" : "benchmarks/finesse/model_eval/model_eval_baseline.zip",
    "finesse-main-0" : "benchmarks/finesse/model_eval/model_eval_split_0.zip",
    "finesse-main-1" : "benchmarks/finesse/model_eval/model_eval_split_1.zip",
    "finesse-main-2" : "benchmarks/finesse/model_eval/model_eval_split_2.zip",
    "finesse-main-3" : "benchmarks/finesse/model_eval/model_eval_split_3.zip",
    "lemb-main" : "benchmarks/lemb/model_eval/model_eval.zip",
    "lemb-baseline": "benchmarks/lemb/model_eval/model_eval_baseline.zip"
}

MV_CHART = {
    "evals/special/srs-16" : "benchmarks/finesse/additional_model_eval/cultura-en-l16-srs-weak",
    "evals/special/rss-wikipedia" : "benchmarks/finesse/additional_model_eval/wikipedia-en-l15-rss",
    "evals/typical/finesse/model_eval/finesse-baseline" : "benchmarks/finesse/model_eval/model_eval_baseline",
    "evals/typical/finesse/model_eval/finesse-main/split_0" : "benchmarks/finesse/model_eval/model_eval_split_0",
    "evals/typical/finesse/model_eval/finesse-main/split_1" : "benchmarks/finesse/model_eval/model_eval_split_1",
    "evals/typical/finesse/model_eval/finesse-main/split_2" : "benchmarks/finesse/model_eval/model_eval_split_2",
    "evals/typical/finesse/model_eval/finesse-main/split_3" : "benchmarks/finesse/model_eval/model_eval_split_3",
    "evals/typical/lemb/model_eval/lemb-main" : "benchmarks/lemb/model_eval/model_eval_2",
    "evals/typical/lemb/model_eval/lemb-baseline": "benchmarks/lemb/model_eval/model_eval_baseline"
}

def unzipper():
    for i, (instruct, path) in enumerate(UNZIP_CHART.items()):
        print(f"Unzipping {instruct} : {i + 1} / {len(UNZIP_CHART)} ...")
        path_unzipped = str(path)[:-4]
        if path_unzipped == "benchmarks/lemb/model_eval/model_eval":
            path_unzipped = "benchmarks/lemb/model_eval/model_eval_2"
        path = Path(path)
        mv_chart_swapped = {v: k for k, v in MV_CHART.items()}
        mv_dir = mv_chart_swapped.get(str(path_unzipped))
        if not (os.path.exists(path_unzipped) or os.path.exists(mv_dir)):
            shutil.unpack_archive(path, path.parent)
            print(f"Unzip Succeed! {instruct} : {i + 1} / {len(UNZIP_CHART)} ...")
        else:
            print(f"Unzip Skipped! {instruct} : {i + 1} / {len(UNZIP_CHART)} ...")

def mv_operator():
    for i, (to, fr) in enumerate(MV_CHART.items()):
        print(f"Moving {fr} -> {to} : {i + 1} / {len(MV_CHART)} ...")
        fr_path = Path(fr)
        to_path = Path(to)
        if not os.path.exists(to_path):
            shutil.move(fr_path, to_path)
            print(f"Mov Succeed! {fr} -> {to} : {i + 1} / {len(MV_CHART)} ...")
        else:
            print(f"Mov Skipped! {fr} -> {to} : {i + 1} / {len(MV_CHART)} ...")


if __name__ == "__main__":
    unzipper()
    mv_operator()