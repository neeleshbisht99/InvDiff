import json
import os
import random

import click


@click.command()
@click.option("--seed", default=0, type=int)
@click.option("--purity", default=1.0, type=float)
def main(purity: float, seed: int):
    random.seed(0)
    root = "data/datasets/pairedimagesets/VisDiffBench"
    easy = [json.loads(line) for line in open(f"{root}/easy.jsonl")]
    medium = [json.loads(line) for line in open(f"{root}/medium.jsonl")]
    hard = [json.loads(line) for line in open(f"{root}/hard.jsonl")]
    data = (
        random.sample(easy, 3) +
        random.sample(medium, 3) +
        random.sample(hard, 4)
    )
    
    for idx in range(0, len(data)):
        item = data[idx]
        cfg = f"""
project: PairedImageSets
seed: {seed}  # random seed

data:
  name: PairedImageSets
  group1: "{item['set1']}"
  group2: "{item['set2']}"
  purity: {purity}
"""

        difficulty = (
            "easy"
            if idx < len(easy)
            else "medium"
            if idx < len(easy) + len(medium)
            else "hard"
        )
        cfg_dir = f"configs/sweep_visdiffbench_purity{purity}_seed{seed}"
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        cfg_file = f"{cfg_dir}/{idx}_{difficulty}.yaml"
        with open(cfg_file, "w") as f:
            f.write(cfg)
        print(f"python main.py --config {cfg_file}")
        os.system(f"python main.py --config {cfg_file}")


if __name__ == "__main__":
    main()
