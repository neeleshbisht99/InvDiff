import pandas as pd

root = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/sweeps/output/CCDiff-MetaShift/"

data = pd.read_csv(root + 'runs.csv')

f = open(root+"output.txt", 'w')

print("Metashift Metrics", file=f)
print("#### CCDiff (Image & Text(CLIP) Evaluator(GPT-4))", file=f)
print("#Group A / Class 0", file=f)
arr0 = [
    ["Dataset ", "acc@1 ", "acc@5 "],
    ["MimicCxrImageSets    ", f"{round(data['Group A/acc@1'].mean(), 2)}   ", f"{round(data['Group A/acc@5'].mean(), 2)}  "]
]

for row in arr0:
    str = "".join(row)
    print(str, file=f)


# python3 invdiff/scripts/metashift_metrics.py