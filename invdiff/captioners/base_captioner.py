import random
from typing import Dict, List, Tuple

from PIL import Image
import wandb

from invdiff.serve.utils_vlm import get_vlm_output

class Captioner:
    def __init__(self, args: Dict):
        self.args = args

    def sample(self, dataset: List[Dict], n: int) -> List[Dict]:
        return random.sample(dataset, n)
    
    def captioning(self, dataset: List[Dict]):
        for item in dataset:
            item["caption"] = get_vlm_output(
                item["path"],
                self.args["captioner"]["prompt"],
                self.args["captioner"]["model"],
            )
    
    def generate_captions(
        self, datasetA: List[Dict], datasetB: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        all_images = []
        random.seed(self.args["seed"])
        for i in range(self.args["num_rounds"]):
            sampled_datasetA = self.sample(datasetA, self.args["num_samples"])
            sampled_datasetB = self.sample(datasetB, self.args["num_samples"])
            self.captioning(sampled_datasetA)
            self.captioning(sampled_datasetB)
            images = self.visualize(sampled_datasetA, sampled_datasetB)
            all_images.append(images)
        return all_images, sampled_datasetA, sampled_datasetB

    def visualize(
        self, sampled_datasetA: List[Dict], sampled_datasetB: List[Dict]
    ) -> Dict:
        imagesA = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_datasetA
        ]
        imagesB = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_datasetB
        ]
        images = {"images_group_A": imagesA, "images_group_B": imagesB}
        return images