import pandas as pd

root = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/sweeps/output/test/"

data = pd.read_csv(root + 'runs.csv')

f = open(root+"output.txt", 'w')

print("Metrics", file=f)
arr1 = [
    ["Dataset ", "acc@1 ", "acc@5 "],
    ["MimicCxrImageSets    ", f"{round(data['acc@1'].mean(), 2)}   ", f"{round(data['acc@5'].mean(), 2)}  "]
]

for row in arr1:
    str = "".join(row)
    print(str, file=f)
