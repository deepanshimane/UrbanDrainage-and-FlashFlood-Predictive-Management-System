# Person B progress tracker (ML model + optimizer)

The AI updates this file after every step. Do not delete the table.

**Overall progress: 13%**
**Last updated:** 2026-10-31
**Right now working on:** B2

| Step | What it is (simple words) | Weight | Status | Result / number |
|---|---|---|---|---|
| B0 | Set up my laptop and check the fake data works | 5 | DONE | 60 storms, 72 steps, 200 nodes, 358 edges; 1.7% flooded |
| B1 | Teach the computer to read the data and cut it into training pieces | 8 | DONE | Input (32, 8, 200, 4), Target (32, 12, 200), split 42/9/9 |
| B2 | Write the "score card" functions (how wrong are we?) | 7 | TODO | |
| B3 | Build 2 simple baseline models to beat | 10 | TODO | |
| B4 | Build and train the main graph AI model (ST-GNN) | 20 | TODO | |
| B5 | Write `predict.py` so Person C can use my model | 10 | TODO | |
| B6 | Make the evaluation report with tables and charts | 8 | TODO | |
| B7 | Optimizer v1: simple pump rules | 8 | TODO | |
| B8 | Optimizer v2: search many schedules, show before/after | 8 | TODO | |
| B9 | Swap in Person A's REAL dataset and retrain | 6 | TODO | |
| B10 | Final numbers for 3 scenarios (light, heavy, extreme) | 5 | TODO | |
| B11 | Test on a second area and write handoff notes for the pitch | 5 | TODO | |

Status values: TODO, IN PROGRESS, DONE, BLOCKED.

## Key numbers (fill in as we go; these go into the pitch)
- Baseline 1 (persistence) flood F1 at +1h / +2h / +3h:
- Baseline 2 (no-graph model) flood F1 at +1h / +2h / +3h:
- ST-GNN flood F1 at +1h / +2h / +3h:
- ST-GNN MAE (cm) at +1h / +2h / +3h:
- Optimizer reduction in flood cost: light / heavy / extreme:

## Blockers and waiting-on
(none yet)

## Notes for the team
(none yet)
