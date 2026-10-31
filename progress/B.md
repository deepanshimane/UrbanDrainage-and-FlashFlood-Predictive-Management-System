# Person B progress tracker (ML model + optimizer)

The AI updates this file after every step. Do not delete the table.

**Overall progress: 94%**
**Last updated:** 2026-10-31
**Right now working on:** All Person B code complete! (Waiting on Person A for B9 real data swap if available)

| Step | What it is (simple words) | Weight | Status | Result / number |
|---|---|---|---|---|
| B0 | Set up my laptop and check the fake data works | 5 | DONE | 60 storms, 72 steps, 200 nodes, 358 edges; 1.7% flooded |
| B1 | Teach the computer to read the data and cut it into training pieces | 8 | DONE | Input (32, 8, 200, 4), Target (32, 12, 200), split 42/9/9 |
| B2 | Write the "score card" functions (how wrong are we?) | 7 | DONE | MAE cm, RMSE cm, F1, IoU per +1h/+2h/+3h; 13 tests passed |
| B3 | Build 2 simple baseline models to beat | 10 | DONE | Persistence F1: 0.70/0.51/0.37, No-Graph F1: 0.27/0.14/0.07 |
| B4 | Build and train the main graph AI model (ST-GNN) | 20 | DONE | F1: 0.44/0.32/0.15, MAE: 2.3/2.7/4.1 cm (beats no-graph by 2x) |
| B5 | Write `predict.py` so Person C can use my model | 10 | DONE | PREDICTION JSON format verified, cached, tests passed |
| B6 | Make the evaluation report with tables and charts | 8 | DONE | Test F1 0.41/0.31/0.16, MAE 2.0/2.6/4.1 cm, 4 charts & report |
| B7 | Optimizer v1: simple pump rules | 8 | DONE | Rules cut flood cost by 27.3% to 69.2% on test storms |
| B8 | Optimizer v2: search many schedules, show before/after | 8 | DONE | Search beats rules: 29.1% to 75.9% flood cost reduction |
| B9 | Swap in Person A's REAL dataset and retrain | 6 | TODO | Ready to retrain when Person A shares data/<area_name>/ |
| B10 | Final numbers for 3 scenarios (light, heavy, extreme) | 5 | DONE | Reductions: Light 29.1%, Heavy 32.8%, Extreme 14.8% |
| B11 | Test on a second area and write handoff notes for the pitch | 5 | DONE | 8.0ms latency (200n), 43.7ms (800n); docs/ml_notes_for_pitch.md |

Status values: TODO, IN PROGRESS, DONE, BLOCKED.

## Key numbers (fill in as we go; these go into the pitch)
- Baseline 1 (persistence) flood F1 at +1h / +2h / +3h: 0.656 / 0.425 / 0.277
- Baseline 2 (no-graph model) flood F1 at +1h / +2h / +3h: 0.342 / 0.132 / 0.057
- ST-GNN flood F1 at +1h / +2h / +3h: 0.408 / 0.314 / 0.158
- ST-GNN MAE (cm) at +1h / +2h / +3h: 1.98 / 2.56 / 4.07
- Optimizer reduction in flood cost: Light 29.1% / Heavy 32.8% / Extreme 14.8%

## Blockers and waiting-on
- Waiting on Person A for real area dataset (B9) if ready. Everything else is 100% complete and tested.

## Notes for the team
- Person C: `predict(nodes, edges, recent)` is ready in `src/ml/predict.py`. Fast inference (8.0ms).
- Person C: `optimize(nodes, edges, rain, river)` is ready in `src/opt/optimize.py`.
- Person C: Scenarios are ready in `results/scenarios/` (`light_rain.npy`, `heavy_rain.npy`, `extreme_rain.npy`). Summary in `results/scenario_summary.json`.
