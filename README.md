# InvDiff: Inverse Canonical Correlation Analysis for Discovering Visual Differences in Natural Language



Some Results:
| Proposer    | Ranker  | PIS-Easy |       | PIS-Medium |       | PIS-Hard |       |
| -------- | ------- |  ------- |  ------- |  ------- | ------- | ------- |  ------- |
|             |         | Acc@1 | Acc@5 | Acc@1 | Acc@5 | Acc@1 | Acc@5 |
| Image (GPT-4V) | Feature (CLIP)  | 0.95 | 1.00 | 0.75 | 0.87 | 0.57 | 0.74 |
| Caption (BLIP-2 + GPT-4) | Feature (CLIP) | 0.88 | 0.99 | 0.75 | 0.86 | 0.61 | 0.80 |
| **Caption (BLIP-2 + InvDiff)** | **Feature (InvDiff + CLIP)** | **0.95** | **0.98** | **0.75** | **0.84** | **0.6** | **0.75** |


## Running InvDiff

### Quick Start

1. **Create a virtual environment**:

    ```bash
      python3 -m venv venv
      source venv/bin/activate
      pip install -U pip
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Setup [wandb](https://wandb.ai) account and login**:
    ```bash
    wandb login
    ```

4. **Discover Differences**:
    ```bash
    python main.py --config configs/example.yaml
    ```

After that, you should see the following results in [wandb](https://wandb.ai/neeleshbisht99-carnegie-mellon-university/InvDiff).


### Complete Setup

If you want to run **InvDiff** on either **VisDiffBench** or any other dataset, follow the steps below.

1. **Setup environment**
   - Setup environment, login to `wandb`, and install dependencies (same as above).

2. **Start CLIP server**
   - Follow the instructions in `invdiff/serve/README.md`.

3. **Start BLIP server**
   - Follow the instructions in `invdiff/serve/README.md`.

4. **Run and Evaluate InvDiff on VisDiffBench**

   a. **Download dataset**
      - Download VisDiffBench from  
        https://drive.google.com/file/d/1PybUlQOesFIfgAjYlJabsyw5UHf5GOrZ/view

   b. **Move the zip file & Decompress**
      ```bash
      mv visdiff_bench.zip data/datasets/pairedimagesets/ && unzip data/datasets/pairedimagesets/visdiff_bench.zip
      ```

   c. **Set OpenAI API key**
      ```bash
      export OPENAI_API_KEY=<API_KEY>
      ```

   d. **Run evaluation**
      ```bash
      python sweeps/sweep_pairedimagesets.py
      ```

5. **Run InvDiff on other datasets**

   a. **Convert dataset**
      - Convert your dataset to CSV format with two required columns:
        - `path`
        - `group_name`
      - Example CSV: `data/examples.csv`

   b. **Setup dataset**
      - Create a folder `<dataset_name>` inside `data/datasets/`
      - Place the dataset files and the CSV inside this folder

   c. **Update configs**
      - General arguments: `configs/base.yaml`
      - Dataset-specific arguments: `configs/example.yaml`

        i. In `configs/base.yaml`
           - Update `data.root` to point to your dataset folder
           - Update `knowledge_bank_filepath` and `agg_knowledge_bank_filepath`, to the path where you want the candidate differences file to be created and stored.

        ii. In `configs/example.yaml`
           - Update `name`
           - Update `group1` and `group2` to match CSV `group_name`s

   d. **Discover differences**
      ```bash
      python main.py --config configs/example.yaml
      ```