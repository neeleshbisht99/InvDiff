import wandb
import pandas as pd

api = wandb.Api(api_key="")

PROJECT = "neeleshbisht99-carnegie-mellon-university/VisDiff-original-30-PairedImageSets"  # replace with your entity/project

runs = api.runs(PROJECT)

records = []
for run in runs:
    try:
        system_metrics = run.history(stream="events")
        
        if system_metrics.empty:
            continue

        gpu_mem_cols = [c for c in system_metrics.columns if "memoryAllocatedBytes" in c]

        record = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
        }

        for col in gpu_mem_cols:
            gpu_id = col.split(".")[1]
            record[f"peak_{gpu_id}_memory_gb"] = system_metrics[col].max() / (1024**3)

        if gpu_mem_cols:
            record["total_peak_memory_gb"] = system_metrics[gpu_mem_cols].max().sum() / (1024**3)

        util_cols = [c for c in system_metrics.columns if "gpu." in c and "utilization" in c.lower()]
        for col in util_cols:
            gpu_id = col.split(".")[1]
            record[f"peak_{gpu_id}_utilization_pct"] = system_metrics[col].max()

        records.append(record)

    except Exception as e:
        print(f"Failed for run {run.id}: {e}")
        continue

df = pd.DataFrame(records)
print(df.to_string())
df.to_csv("wandb_system_metrics.csv", index=False)
print("Saved to wandb_system_metrics.csv")


# python3 invdiff/scripts/metrics/gpu_mem/gpu_usage.py