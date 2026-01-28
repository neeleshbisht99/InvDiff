import json
import os
import random

import click


@click.command()
@click.option("--seed", default=0, type=int)
@click.option("--purity", default=1.0, type=float)
def main(purity: float, seed: int):
    random.seed(0)
    root = "data/datasets/metashift"
    data = [json.loads(line) for line in open(f"{root}/pairedsets.jsonl")]

    for idx in range(0, 11):
        item = data[idx]
        cfg = f"""
project: MetaShift
seed: {seed}  # random seed

data:
  name: MetaShift
  group1: "{item['set1']}"
  group2: "{item['set2']}"
  purity: {purity}
"""

        cfg_dir = f"configs/sweep_metashift_purity{purity}_seed{seed}"
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        cfg_file = f"{cfg_dir}/{idx}.yaml"
        with open(cfg_file, "w") as f:
            f.write(cfg)
        print(f"python main.py --config {cfg_file}")
        os.system(f"python main.py --config {cfg_file}")


if __name__ == "__main__":
    main()
