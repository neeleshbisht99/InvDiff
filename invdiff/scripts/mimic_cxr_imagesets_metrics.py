import pandas as pd

root = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/sweeps/output/MimicCxrImageSets-domain-kb-exp-run1/"

df = pd.read_csv(root + 'runs.csv')

easy_df = df[df["config"].str.contains("easy.yaml", case=False, na=False)]
medium_df = df[df["config"].str.contains("medium.yaml", case=False, na=False)]
hard_df = df[df["config"].str.contains("hard.yaml", case=False, na=False)]


f = open(root+"output.txt", 'w')

print("Metrics", file=f)
print("#### CCDiff (MIMIC-CXR)", file=f)

arr0 = [
    ["Dataset ", "acc@1 ", "acc@5 "],
    ["Easy    ", f"{round(easy_df['Group A/acc@1'].mean(), 2)}   ", f"{round(easy_df['Group A/acc@5'].mean(), 2)}  "],
    ["Medium  ", f"{round(medium_df['Group A/acc@1'].mean(), 2)}   ", f"{round(medium_df['Group A/acc@5'].mean(), 2)}  "],
    ["Hard    ", f"{round(hard_df['Group A/acc@1'].mean(), 2)}   ", f"{round(hard_df['Group A/acc@5'].mean(), 2)}  "]
]

for row in arr0:
    str = "".join(row)
    print(str, file=f)


# python3 invdiff/scripts/mimic_cxr_imagesets_metrics.py
