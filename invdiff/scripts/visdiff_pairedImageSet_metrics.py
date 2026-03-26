import pandas as pd

root = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/sweeps/output/runtime/Visdiff-original-30-pairedimagesets/"

df = pd.read_csv(root + 'runs.csv')

# easy_df = pd.read_csv(root + 'easy_runs.csv')
# medium_df = pd.read_csv(root + 'medium_runs.csv')
# hard_df = pd.read_csv(root + 'hard_runs.csv')

easy_df = df[df["config"].str.contains("easy.yaml", case=False, na=False)]
medium_df = df[df["config"].str.contains("medium.yaml", case=False, na=False)]
hard_df = df[df["config"].str.contains("hard.yaml", case=False, na=False)]


f = open(root+"output.txt", 'w')

print("Metrics", file=f)
print("#### VisDiff", file=f)
arr0 = [
    ["Dataset ", "acc@1 ", "acc@5 "],
    ["Easy    ", f"{round(easy_df['acc@1'].mean(), 2)}   ", f"{round(easy_df['acc@5'].mean(), 2)}  "],
    ["Medium  ", f"{round(medium_df['acc@1'].mean(), 2)}   ", f"{round(medium_df['acc@5'].mean(), 2)}  "],
    ["Hard    ", f"{round(hard_df['acc@1'].mean(), 2)}   ", f"{round(hard_df['acc@5'].mean(), 2)}  "]
]

for row in arr0:
    str = "".join(row)
    print(str, file=f)

print("\n", file=f)
print(f"#Avg. Runtime {round(df['Runtime'].mean(), 2)}", file=f)

# python3 invdiff/scripts/visdiff_pairedImageSet_metrics.py