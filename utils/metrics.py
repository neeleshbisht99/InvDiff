import pandas as pd

root = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/utils/"

easy_df = pd.read_csv(root + 'easy_runs.csv')
medium_df = pd.read_csv(root + 'medium_runs.csv')
hard_df = pd.read_csv(root + 'hard_runs.csv')

print("Metrics")
print("#### Anti-CCA (Image & Text(BLIP-2[smaller-variant] + CLIP) Evaluator(GPT-4))")
print("#Group A / Class 0")
arr0 = [
    ["Dataset ", "acc@1 ", "acc@5 "],
    ["Easy    ", f"{round(easy_df['Group A/acc@1'].mean(), 2)}   ", f"{round(easy_df['Group A/acc@5'].mean(), 2)}  "],
    ["Medium  ", f"{round(medium_df['Group A/acc@1'].mean(), 2)}   ", f"{round(medium_df['Group A/acc@5'].mean(), 2)}  "],
    ["Hard    ", f"{round(hard_df['Group A/acc@1'].mean(), 2)}   ", f"{round(hard_df['Group A/acc@5'].mean(), 2)}  "]
]

for row in arr0:
    str = "".join(row)
    print(str)

print("\n")
print("### Anti-CCA (Image & Text(BLIP-2[smaller-variant] + CLIP) Evaluator(GPT-4))")
print("#Group B / Class 1")
arr1 = [
    ["Dataset ", "acc@1 ", "acc@5 "],
    ["Easy    ", f"{round(easy_df['Group B/acc@1'].mean(), 2)}   ", f"{round(easy_df['Group B/acc@5'].mean(), 2)}  "],
    ["Medium  ", f"{round(medium_df['Group B/acc@1'].mean(), 2)}   ", f"{round(medium_df['Group B/acc@5'].mean(), 2)}  "],
    ["Hard    ", f"{round(hard_df['Group B/acc@1'].mean(), 2)}   ", f"{round(hard_df['Group B/acc@5'].mean(), 2)}  "]
]

for row in arr1:
    str = "".join(row)
    print(str)
