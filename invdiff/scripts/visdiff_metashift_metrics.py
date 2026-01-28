import pandas as pd

root = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/sweeps/output/VisDiff-Metashift-run1/"

data = pd.read_csv(root + 'runs.csv')

f = open(root + "output.txt", 'w')

print("Metrics", file=f)
print("#### VisDiff (Image & Text(BLIP-2 + GPT-4) Evaluator(GPT-4))", file=f)
arr0 = [
    ["Dataset ", "acc@1 ", "acc@5 "],
    ["Metashift    ", f"{round(data['acc@1'].mean(), 2)}   ", f"{round(data['acc@5'].mean(), 2)}  "],
]

for row in arr0:
    str = "".join(row)
    print(str, file=f)
