import pandas as pd

root = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/sweeps/output/VisDiff-MimicCxrImageSets-cxr-run4/"

easy_df = pd.read_csv(root + 'easy_runs.csv')
medium_df = pd.read_csv(root + 'medium_runs.csv')
hard_df = pd.read_csv(root + 'hard_runs.csv')

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


# data = pd.read_csv(root + 'runs.csv')

# f = open(root+"output.txt", 'w')

# print("Metrics", file=f)
# arr1 = [
#     ["Dataset ", "acc@1 ", "acc@5 "],
#     ["MimicCxrImageSets    ", f"{round(data['acc@1'].mean(), 2)}   ", f"{round(data['acc@5'].mean(), 2)}  "]
# ]

# for row in arr1:
#     str = "".join(row)
#     print(str, file=f)
