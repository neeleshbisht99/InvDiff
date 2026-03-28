import random
from typing import Dict, List, Tuple

from invdiff.serve.utils_vlm import get_vlm_output

class Captioner:
    def __init__(self, args: Dict):
        self.args = args
        self.analysis_type = self.args.get("analysis", "full")

    def sample(self, dataset: List[Dict], n: int) -> List[Dict]:
        return random.sample(dataset, n)
    
    def captioning(self, dataset: List[Dict]):
        for item in dataset:
            item["caption"] = get_vlm_output(
                item["path"],
                self.args["prompt"],
                self.args["model"],
            )
    
    def generate_captions(
        self, datasetA: List[Dict], datasetB: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        random.seed(self.args["seed"])
        for _ in range(self.args["num_rounds"]):
            sampled_datasetA = self.sample(datasetA, self.args["num_samples"])
            self.captioning(sampled_datasetA)
            sampled_datasetB = None
            if self.analysis_type == "full":
                sampled_datasetB = self.sample(datasetB, self.args["num_samples"])
                self.captioning(sampled_datasetB)
        return sampled_datasetA, sampled_datasetB