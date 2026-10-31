# Person B progress tracker (ML model + optimizer)

The AI updates this file after every step. Do not delete the table.

**Overall progress: 60%**
**Last updated:** 2026-10-31
**Right now working on:** B6

| Step | What it is (simple words) | Weight | Status | Result / number |
|---|---|---|---|---|
| B0 | Set up my laptop and check the fake data works | 5 | DONE | 60 storms, 72 steps, 200 nodes, 358 edges; 1.7% flooded |
| B1 | Teach the computer to read the data and cut it into training pieces | 8 | DONE | Input (32, 8, 200, 4), Target (32, 12, 200), split 42/9/9 |
| B2 | Write the "score card" functions (how wrong are we?) | 7 | DONE | MAE cm, RMSE cm, F1, IoU per +1h/+2h/+3h; 13 tests passed |
| B3 | Build 2 simple baseline models to beat | 10 | DONE | Persistence F1: 0.70/0.51/0.37, No-Graph F1: 0.27/0.14/0.07 |
| B4 | Build and train the main graph AI model (ST-GNN) | 20 | DONE | F1: 0.44/0.32/0.15, MAE: 2.3/2.7/4.1 cm (beats no-graph by 2x) |
| B5 | Write `predict.py` so Person C can use my model | 10 | DONE | PREDICTION JSON format verified, cached, tests passed |
| B6 | Make the evaluation report with tables and charts | 8 | TODO | |
| B7 | Optimizer v1: simple pump rules | 8 | TODO | |
| B8 | Optimizer v2: search many schedules, show before/after | 8 | TODO | |
| B9 | Swap in Person A's REAL dataset and retrain | 6 | TODO | |
| B10 | Final numbers for 3 scenarios (light, heavy, extreme) | 5 | TODO | |
| B11 | Test on a second area and write handoff notes for the pitch | 5 | TODO | |

Status values: TODO, IN PROGRESS, DONE, BLOCKED.

## Key numbers (fill in as we go; these go into the pitch)
- Baseline 1 (persistence) flood F1 at +1h / +2h / +3h: 0.701 / 0.505 / 0.367
- Baseline 2 (no-graph model) flood F1 at +1h / +2h / +3h: 0.269 / 0.144 / 0.068
- ST-GNN flood F1 at +1h / +2h / +3h: 0.437 / 0.323 / 0.147
- ST-GNN MAE (cm) at +1h / +2h / +3h: 2.27 / 2.72 / 4.12
- Optimizer reduction in flood cost: light / heavy / extreme:

## Blockers and waiting-on
(none yet)

## Notes for the team
(none yet)
