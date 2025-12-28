import logging
from typing import Dict, List, Tuple

import click
import pandas as pd
from omegaconf import OmegaConf
import json
import os
import random

import wandb
from invdiff.captioners.base_captioner import Captioner

def load_config(config: str) -> Dict:
    base_cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    args = OmegaConf.to_container(final_cfg)
    args["config"] = config
    if args["wandb"]:
        wandb.init(
            project=args["project"],
            name=args["data"]["name"],
            group=f'{args["data"]["group1"]} - {args["data"]["group2"]} ({args["data"]["purity"]})',
            config=args,
        )
    return args

def load_data(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"]

    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv")

    if data_args["subset"]:
        old_len = len(df)
        df = df[df["subset"] == data_args["subset"]]
        print(
            f"Taking {data_args['subset']} subset (dataset size reduced from {old_len} to {len(df)})"
        )

    datasetA = df[df["group_name"] == data_args["group1"]].to_dict("records")
    datasetB = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]

    if data_args["purity"] < 1:
        logging.warning(f"Purity is set to {data_args['purity']}. Swapping groups.")
        assert len(datasetA) == len(datasetB), "Groups must be of equal size"
        n_swap = int((1 - data_args["purity"]) * len(datasetA))
        datasetA = datasetA[n_swap:] + datasetB[:n_swap]
        datasetB = datasetB[n_swap:] + datasetA[:n_swap]
    return datasetA, datasetB, group_names


def generate_captions(args: Dict, datasetA: List[Dict], datasetB: List[Dict]) -> List[str]:
    proposer_args = args["captioner"]
    proposer_args["seed"] = args["seed"]
    proposer_args["captioner"] = args["captioner"]

    captioner = eval(proposer_args["method"])(proposer_args)
    images, sampled_dataset1, sampled_dataset2 = captioner.get_captions(datasetA, datasetB)
    if args["wandb"]:
        for i in range(len(images)):
            wandb.log(
                {
                    f"group 1 images ({datasetA[0]['group_name']})": images[i][
                        "images_group_1"
                    ],
                    f"group 2 images ({datasetB[0]['group_name']})": images[i][
                        "images_group_2"
                    ],
                }
            )
    return sampled_dataset1, sampled_dataset2

def prepare_knowledge_bank(args: Dict, datasetA: List[Dict], datasetB: List[Dict], group_names:List[str]):
    captioner_args = args["captioner"]
    filename = captioner_args["knowledge_bank_filepath"]
    a = group_names[0]
    b = group_names[1]
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump({}, file, indent=4)

    with open(filename, 'r') as file:
        data = json.load(file)
    
    data[a] = [item['caption'].replace("\n", " ").strip() for item in datasetA]
    data[b] = [item['caption'].replace("\n", " ").strip() for item in datasetB]

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    logging.info(f"Saved {len(data[a])} captions for {a}")
    logging.info(f"Saved {len(data[b])} captions for {b}")

def prepare_agg_knowledge_bank(args: Dict):
    captioner_args = args["captioner"]
    knowledge_bank_filepath = captioner_args["knowledge_bank_filepath"]
    hypo_data = {}
    with open(knowledge_bank_filepath, 'r') as file:
        hypo_data = json.load(file)

    knowledge_bank = []
    random.seed(0)
    for k, v in hypo_data.items():
        knowledge_bank.extend(v)
    random.shuffle(knowledge_bank)

    agg_knowledge_bank_filepath = captioner_args["agg_knowledge_bank_filepath"]
    with open(agg_knowledge_bank_filepath, 'w') as file:
        json.dump(knowledge_bank, file, indent=4)


@click.command()
@click.option("--config", help="config file")
def main(config):
    logging.info("Loading config...")
    args = load_config(config)
    logging.info("Loading data...")
    datasetA, datasetB, group_names = load_data(args)

    captioned_dataset1, captioned_dataset2 = generate_captions(args, datasetA, datasetB)
    prepare_knowledge_bank(args, captioned_dataset1, captioned_dataset2, group_names)


if __name__ == "__main__":
    main()