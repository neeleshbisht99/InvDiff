import pandas as pd

ccDiff_file = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/invdiff/scripts/metrics/runtime/ccdiff_original.csv"

visDiff_file = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/invdiff/scripts/metrics/runtime/visdiff_original.csv"

ccDiff_df = pd.read_csv(ccDiff_file)
visDiff_df = pd.read_csv(visDiff_file)


print("VisDiff runtime: ", visDiff_df["Runtime"].astype(int).sum())
print("CCDiff runtime: ", ccDiff_df["Runtime"].astype(int).sum())

# VisDiff runtime:  5293
# CCDiff runtime:  5518