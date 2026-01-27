import json

# Only ran for 60 items
f_path = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/invdiff/scripts/metrics/tokens/visdiff_original.json"
with open(f_path, 'r') as f:
    data = json.load(f)


prompt_tokens = 0
completion_tokens = 0
total_tokens = 0

for item in data:
    prompt_tokens += item["usage"]["prompt_tokens"]
    completion_tokens += item["usage"]["completion_tokens"]
    total_tokens += item["usage"]["total_tokens"]

print("prompt_tokens ",prompt_tokens)
print("completion_tokens ",completion_tokens)
print("total_tokens ",total_tokens)

# prompt_tokens  184715
# completion_tokens  14835
# total_tokens  199550

# 60 -> 184715
# 60 -> 14835
# 60 -> 199550

# 150 -> ( 184715 * 150 ) / 60 =  461787.5
# 150 -> ( 14835 * 150 ) / 60 = 37087.5
# 150 -> ( 199550 * 150 ) / 60 = 498875

# 461787.5 + 37087.5 = 498875 = 0.5 million