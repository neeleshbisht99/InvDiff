# InvDiff: Inverse Canonical Correlation Analysis for Discovering Visual Differences in Natural Language



Some Results:
| Proposer    | Ranker  | PIS-Easy |       | PIS-Medium |       | PIS-Hard |       |
| -------- | ------- |  ------- |  ------- |  ------- | ------- | ------- |  ------- |
|             |         | Acc@1 | Acc@5 | Acc@1 | Acc@5 | Acc@1 | Acc@5 |
| Image (GPT-4V) | Feature (CLIP)  | 0.95 | 1.00 | 0.75 | 0.87 | 0.57 | 0.74 |
| Caption (BLIP-2 + GPT-4) | Feature (CLIP) | 0.88 | 0.99 | 0.75 | 0.86 | 0.61 | 0.80 |
| **Caption (BLIP-2 + InvDiff)** | **Feature (InvDiff + CLIP)** | **0.95** | **0.98** | **0.75** | **0.84** | **0.6** | **0.75** |

