import logging
import random
import time
from typing import Dict, List, Tuple
from pathlib import Path

import click
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
import json
import os

import wandb
from invdiff.evaluators.base_evaluator import Evaluator
from invdiff.captioners.base_captioner import Captioner
from invdiff.invdiff import InvDiff

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

    dataset1 = df[df["group_name"] == data_args["group1"]].to_dict("records")
    dataset2 = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]

    if data_args["purity"] < 1:
        logging.warning(f"Purity is set to {data_args['purity']}. Swapping groups.")
        assert len(dataset1) == len(dataset2), "Groups must be of equal size"
        n_swap = int((1 - data_args["purity"]) * len(dataset1))
        dataset1 = dataset1[n_swap:] + dataset2[:n_swap]
        dataset2 = dataset2[n_swap:] + dataset1[:n_swap]
    return dataset1, dataset2, group_names

def compute_differences(
    args: Dict,
    dataset1: List[Dict],
    dataset2: List[Dict]
) -> List[str]:
    """
    Discover distinctive features for each dataset using InvDiff.
    
    Returns:
        differences_A: Features distinctive to dataset A (present in A, absent in B)
        classname_A: Name of dataset/class A
        differences_B: Features distinctive to dataset B (present in B, absent in A)  
        classname_B: Name of dataset/class B
    """
    inv_diff_args = args["inv_diff"]
    seed = args["seed"]
    inv_diff = eval(inv_diff_args["method"])(inv_diff_args)

    differences_A, classname_A, differences_B, classname_B, exec_time_logs = inv_diff.get_differences(dataset1, dataset2, seed)
    # if args["wandb"]:
    #     table_A = wandb.Table(dataframe=pd.DataFrame(differences_A))
    #     wandb.log({f"Scored Differences ({classname_A} > {classname_B})": table_A})

    #     table_B = wandb.Table(dataframe=pd.DataFrame(differences_B))
    #     wandb.log({f"Scored Differences ({classname_B} > {classname_A})": table_B})

    differences_A = [diff["text"] for diff in differences_A]
    differences_B = [diff["text"] for diff in differences_B]
    return differences_A, classname_A, differences_B, classname_B, exec_time_logs


def evaluate(args: Dict, differences_A: List[str], classname_A: str, differences_B: List[str], classname_B: str) -> Dict:
    """
    Evaluate discovered differences using an evaluator (e.g., GPT-4).
    
    Args:
        args: Configuration dictionary
        differences_A: Ranked list of differences distinctive to group A
        classname_A: Name of group A
        differences_B: Ranked list of differences distinctive to group B
        classname_B: Name of group B
        
    Returns:
        metrics_A: Evaluation metrics for group A differences
        metrics_B: Evaluation metrics for group B differences
    """
    analysis_type = args.get("analysis", "full")
    evaluator_args = args["evaluator"]
    evaluator = eval(evaluator_args["method"])(evaluator_args)
    metrics_A, eval_A = evaluator.evaluate(
        differences_A,
        classname_A,
        classname_B
    )

    metrics_B = None
    eval_B = None
    if analysis_type == "full":
        metrics_B, eval_B = evaluator.evaluate(
            differences_B,
            classname_B,
            classname_A
        )

    # if args["wandb"] and evaluator_args["method"] != "NullEvaluator":
    #     wandb.log({
    #         # Row 1 (Group A)
    #         "Group A/acc@1": metrics_A["acc@1"],
    #         "Group A/acc@5": metrics_A["acc@5"],
    #         "Group A/acc@N": metrics_A["acc@N"],
    #     })
    #     if metrics_B:
    #         wandb.log({
    #             # Row 2 (Group B)
    #             "Group B/acc@1": metrics_B["acc@1"],
    #             "Group B/acc@5": metrics_B["acc@5"],
    #             "Group B/acc@N": metrics_B["acc@N"],
    #         })
    #     table_A = wandb.Table(dataframe=pd.DataFrame(eval_A))
    #     wandb.log({f"Evaluated Differences ({classname_A} > {classname_B})": table_A})
    #     if eval_B:
    #         table_B = wandb.Table(dataframe=pd.DataFrame(eval_B))
    #         wandb.log({f"Evaluated Differences ({classname_B} > {classname_A})": table_B})

    return metrics_A, metrics_B


# def visualize_dataset(dataset1: List[Dict], dataset2: List[Dict], group_names):
#     images_a = [
#             wandb.Image(
#                 Image.open(item["path"]).convert("RGB").resize((224, 224))
#             )
#             for item in dataset1
#         ]
#     images_b = [
#             wandb.Image(
#                 Image.open(item["path"]).convert("RGB").resize((224, 224))
#             )
#             for item in dataset2
#         ]
#     wandb.log(
#         {
#             f"group A images ({group_names[0]})": images_a,
#             f"group B images ({group_names[1]})": images_b
#         }
#     )

def generate_captions(args: Dict, dataset1: List[Dict], dataset2: List[Dict]) -> List[str]:
    captioner_args = args["captioner"]
    captioner_args["seed"] = args["seed"]
    
    captioner = eval(captioner_args["method"])(captioner_args)
    sampled_dataset1, sampled_dataset2 = captioner.generate_captions(dataset1, dataset2)
    return sampled_dataset1, sampled_dataset2

def prepare_knowledge_bank(args: Dict, dataset1: List[Dict], dataset2: List[Dict], group_names:List[str]):
    captioner_args = args["captioner"]
    filepath = captioner_args["knowledge_bank_filepath"]
    filename = Path(filepath)
    a = group_names[0]
    b = group_names[1]
    if not os.path.exists(filename):
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as file:
            json.dump({}, file, indent=4)

    with open(filename, 'r') as file:
        data = json.load(file)
    
    data[a] = [item['caption'].replace("\n", " ").strip() for item in dataset1]
    data[b] = [item['caption'].replace("\n", " ").strip() for item in dataset2]

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    logging.info(f"Saved {len(data[a])} captions for {a}")
    logging.info(f"Saved {len(data[b])} captions for {b}")

def prepare_agg_knowledge_bank(args: Dict):
    captioner_args = args["captioner"]
    knowledge_bank_filepath = captioner_args["knowledge_bank_filepath"]
    data = {}
    with open(knowledge_bank_filepath, 'r') as file:
        data = json.load(file)

    knowledge_bank = []
    for k, v in data.items():
        knowledge_bank.extend(v)
    random.shuffle(knowledge_bank)

    agg_knowledge_bank_filepath = captioner_args["agg_knowledge_bank_filepath"]
    with open(agg_knowledge_bank_filepath, 'w') as file:
        json.dump(knowledge_bank, file, indent=4)

def create_knowledge_bank(args: Dict, dataset1: List[Dict], dataset2: List[Dict], group_names):
    captioner_args = args["captioner"]
    knowledge_bank_filepath = captioner_args["knowledge_bank_filepath"]
    a = group_names[0]
    b = group_names[1]
    if os.path.exists(knowledge_bank_filepath):
        with open(knowledge_bank_filepath, 'r') as file:
            data = json.load(file)
            if a in data and b in data:
                logging.info("Knowledge bank already exists...")
                return
    captioned_dataset1, captioned_dataset2 = generate_captions(args, dataset1, dataset2)
    prepare_knowledge_bank(args, captioned_dataset1, captioned_dataset2, group_names)
    prepare_agg_knowledge_bank(args)

@click.command()
@click.option("--config", help="config file")
def main(config):
    total_start_time = time.time()
    load_config_and_data_start_time = time.time()
    logging.info("Loading config...")
    args = load_config(config)
    logging.info("Loading data...")
    dataset1, dataset2, group_names = load_data(args)
    load_config_and_data_elapsed = time.time() - load_config_and_data_start_time

    generate_captions_start_time = time.time()
    captioned_dataset1, captioned_dataset2 = generate_captions(args, dataset1, dataset2)
    generate_captions_elapsed = time.time() - generate_captions_start_time
    # logging.info("Creating knowledge bank...")
    # create_knowledge_bank(args, dataset1, dataset2, group_names)
    # if args["wandb"]:
    #     visualize_dataset(dataset1, dataset2, group_names)
    propose_rank_start_time = time.time()
    logging.info("Proposing & Ranking differences...")
    differences_A, classname_A, differences_B, classname_B, exec_time_logs = compute_differences(args, dataset1, dataset2)
    propose_rank_elapsed = time.time() - propose_rank_start_time
    total_elapsed = time.time() - total_start_time
    if args["wandb"]:
        exec_time_logs_dict = {}
        for k, v in exec_time_logs.items():
            exec_time_logs_dict[f"time/propose_rank_{k}_seconds"] = v
            exec_time_logs_dict[f"time/propose_rank_{k}_minutes"] = v / 60.0
        wandb.log({
            "time/total_runtime_seconds": total_elapsed,
            "time/total_runtime_minutes": total_elapsed / 60.0,
            "time/load_config_and_data_seconds": load_config_and_data_elapsed,
            "time/load_config_and_data_minutes": load_config_and_data_elapsed / 60.0,
            "time/generate_captions_seconds": generate_captions_elapsed,
            "time/generate_captions_minutes": generate_captions_elapsed / 60.0,
            "time/total_propose_rank_seconds": propose_rank_elapsed,
            "time/total_propose_rank_minutes": propose_rank_elapsed / 60.0,
            **exec_time_logs_dict
        })
    logging.info(f"Time till evaluation: {elapsed:.2f}s")
    logging.info("Evaluating differences...")
    metrics_A, metrics_B = evaluate(args, differences_A, classname_A, differences_B, classname_B)
    logging.info("Evaluation Completed!")

if __name__ == "__main__":
    main()
