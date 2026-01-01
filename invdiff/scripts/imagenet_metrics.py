import pandas as pd

root = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/datasets/pairedimagesets/output/imagenet/"

imagenetr = pd.read_csv(root + 'imagenetr.csv')
imagenet_star = pd.read_csv(root + 'imagenetstar.csv')

f = open(root+"output.txt", 'w')

print("Metrics", file=f)
print("#### InvDiff (Image & Text(BLIP-2[smaller-variant] + CLIP) Evaluator(GPT-4))", file=f)
print("#Group A / Class 0", file=f)
arr0 = [
    ["Dataset ", "acc@1 ", "acc@5 "],
    ["ImageNet-R    ", f"{round(imagenetr['Group A/acc@1'].mean(), 2)}   ", f"{round(imagenetr['Group A/acc@5'].mean(), 2)}  "],
    ["ImageNet-*    ", f"{round(imagenet_star['Group A/acc@1'].mean(), 2)}   ", f"{round(imagenet_star['Group A/acc@5'].mean(), 2)}  "],
]

for row in arr0:
    str = "".join(row)
    print(str, file=f)


print("\n#Group B / Class 1", file=f)
arr1 = [
    ["Dataset ", "acc@1 ", "acc@5 "],
    ["ImageNet-R    ", f"{round(imagenetr['Group B/acc@1'].mean(), 2)}   ", f"{round(imagenetr['Group B/acc@5'].mean(), 2)}  "],
    ["ImageNet-*    ", f"{round(imagenet_star['Group B/acc@1'].mean(), 2)}   ", f"{round(imagenet_star['Group B/acc@5'].mean(), 2)}  "],
]

for row in arr1:
    str = "".join(row)
    print(str, file=f)
