import pandas as pd
import wandb

api = wandb.Api()
entity, project = "neeleshbisht99-carnegie-mellon-university", "CCDiff-MetaShift"
runs = api.runs(entity + "/" + project)

rows = []
for run in runs:
    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    data = config.get("data", {})
    
    rows.append({
        "Name": config.get("project", ""),
        "Runtime": round(summary.get("_runtime", 0)),
        "analysis": summary.get("analysis", ""),
        "config": config.get("config", ""),
        "data.group1": data.get("group1", ""),
        "data.group2": data.get("group2", ""),
        "Group A/acc@1": summary.get("Group A/acc@1", ""),
        "Group A/acc@5": summary.get("Group A/acc@5", ""),
        "Group A/acc@N": summary.get("Group A/acc@N", ""),
        "time/propose_rank_minutes": summary.get("time/propose_rank_minutes", ""),
        "time/propose_rank_seconds": summary.get("time/propose_rank_seconds", ""),
    })
    print("fetch another set: ", len(rows))

df = pd.DataFrame(rows)
df.to_csv("runs.csv", index=False)
print(f"Total runs: {len(df)}")
print("runs.csv written!")


# python3 invdiff/scripts/wandb_fetch_all_runs.py