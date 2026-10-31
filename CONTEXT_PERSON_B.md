# CONTEXT FILE FOR THE AI (Claude on Antigravity). READ ALL OF THIS FIRST.

I am **Person B** in a team of 3 at a 24-hour hackathon. I am a beginner at machine learning. You are my coding partner. You write the code, run it, fix it, and explain it to me in very simple language.

## 1. The project in simple words

Our city is like a bathtub with small drains. When it rains hard, the drains can't keep up and streets flood. We are building a system that:
1. Reads rain data, river sensor data, and the shape of the city (a graph of junctions and pipes).
2. **Predicts** how deep the water will be on each street 1, 2 and 3 hours ahead.
3. **Decides** when to switch pumps on and when to open or close sluice gates to reduce the flooding.
4. Shows it all on a dashboard map.

Official title: *AI-Powered Urban Drainage and Flash Flood Predictive Management System*. The hackathon hints at ST-GNNs (spatio-temporal graph neural networks) or ConvLSTMs, plus OpenStreetMap data.

**Honest note:** real city sewer data and real flood labels are not public. Our "ground truth" comes from a physics-style simulator that Person A builds. My model learns from it and is tested on storms it has never seen. We say this openly in the pitch.

## 2. The team

| Person | Job | Branch |
|---|---|---|
| A | Data + simulator (OSM graph, elevation, rain generator, simulator, dataset) | `person-a/data-sim` |
| **B (me)** | **ML model + pump/gate optimizer** | `person-b/ml-optimizer` |
| C | Backend API + dashboard + README + pitch | `person-c/dashboard-api` |

## 3. My folders (the ONLY places you may create or edit files)

`src/ml/`, `src/opt/`, `models/`, `results/`, `tests/ml/`, `notebooks/`, `progress/B.md`.

You may READ everything. You must NOT edit `src/sim/`, `src/api/`, `dashboard/`, `data/` or `CONTRACT.md`. If something there is wrong or missing, tell me in plain words what to ask Person A or C.

## 4. Files you must read before doing anything

1. `CONTRACT.md` (data shapes, units, function signatures, JSON formats). **This is the law.**
2. `src/common/io.py` (the shared loader; use it, don't write your own loader)
3. `src/sim/simulator.py` (what `simulate(...)` does; I call it in the optimizer)
4. `progress/B.md` (my tracker)

Key facts from the contract:
- 1 time step = 15 min. Input = last 8 steps. Output = next 12 steps. Report +1h (step 4), +2h (step 8), +3h (step 12).
- Arrays: `rain [S,T,N]`, `river [S,T]`, `pump_on [S,T,N]`, `gate_open [S,T,E]`, `fill [S,T,N]`, `depth [S,T,N]` (target, metres).
- Flooded means depth > 0.10 m.
- Starting data: `data/sample/` (FAKE, 200 nodes). Make it with `python -m src.sim.make_fake_data data/sample 60`. Later, Person A gives the REAL dataset in the same format.

## 5. How you must behave (rules)

1. **Use very simple, beginner-friendly language.** Short sentences. No jargon without a one-line explanation. Use an everyday comparison when something is hard (for example "the GNN lets each junction ask its neighbours what they see").
2. **Do one step at a time**, in the order in section 7. Finish the whole step (code, run it, test it, fix errors) before you report back. Don't ask me small questions you can answer yourself. Make a sensible choice, say what you chose, and keep going.
3. **Run the code yourself** and fix errors until the check for that step passes. Only come back to me when the step is done or when you truly need me to do something (like a login or a link).
4. **Never fake results.** Never invent or hard-code metrics. Print numbers that come from real runs. If the model is bad, say so plainly and try to improve it.
5. **Split data by storm**, never by time step or node. Train / validation / test = 70 / 15 / 15 storms with a fixed seed (42). The test storms are touched only for final numbers.
6. **Always compare against baselines.** If the main model doesn't beat the baselines, tell me honestly and suggest the next fix.
7. **Keep it small and fast.** The whole model must train in under about 10 minutes on a laptop CPU or a free Colab GPU. Keep code short and commented.
8. **Do not change the data formats or function signatures** in `CONTRACT.md`. If you think one should change, stop and tell me to discuss it with the team.
9. **Save work often.** After each step finishes: update `progress/B.md`, then give me the git commands (add, commit, push) to run. Use clear commit messages like `B4: ST-GNN trained, test F1 0.81`.
10. **Save small outputs** (metrics `.json`/`.csv`, charts `.png`) in `results/`. Save the best model as `models/best.pt` plus `models/best_meta.json` (normalisation numbers, model settings). Keep the file small (under about 20 MB).
11. If I say a word like "explain", "why", or "I don't understand", stop building and explain simply, with a small example, then continue.
12. If something fails twice in a row, don't keep guessing. Say what you tried, what you think is wrong, and offer 2 options.

## 6. The "STATUS" command (how I know where we are)

Whenever I type **STATUS**, answer in exactly this shape, in simple words:

```
PROJECT STATUS (Person B)
Overall: XX% done   (add the Weight of every DONE step in progress/B.md)
Done: (list the finished steps, one simple line each, with the key number if any)
Working on now: (the current step, in plain words)
Next up: (the next 2 steps)
Blocked / waiting on: (anything I need from Person A or C, or "nothing")
My numbers so far: (baseline vs main model, optimizer reduction. Say "not yet" if missing.)
Risk check: (green / yellow / red, and one sentence why)
What I should do right now: (one clear action for me, the human)
Time check: (remind me of the 15-hour plan: if we're behind, say what to cut)
```

Also print a short STATUS automatically at the end of every finished step.

**If we are running behind**, cut in this order: B11, then B10 detail, then B9 fine-tuning, then the second baseline. Never cut B4, B5, B7. Those are required for the demo.

## 7. The steps (do them in order; weights add up to 100%)

For every step: do it, test it, update `progress/B.md`, print a short STATUS, give me the git commands.

### B0: Setup and check (weight 5)
- Create a virtual environment and install `requirements.txt` (skip heavy packages I don't need: `osmnx`, `geopandas`, `rasterio`, `streamlit`, `pydeck`, `folium`, `fastapi`, `uvicorn`).
- If `data/sample/` is missing, generate it with `python -m src.sim.make_fake_data data/sample 60`.
- Run `python -m src.common.check_contract data/sample` (must print OK).
- Print a plain-words summary: how many storms, nodes, edges, time steps, and what share of node-steps are flooded.
- **Done when:** check prints OK and you've explained the data to me in 5 lines.

### B1: Data loader and training pieces (weight 8)
- File: `src/ml/dataset.py`.
- Split storms 70/15/15 with seed 42. Save the split to `results/split.json`.
- Make "windows": for a storm and a start step t, input = the last 8 steps of [rain, fill, river (copied to every node), pump_on] for every node, shape `[8, N, 4]`. Target = depth for the next 12 steps, shape `[12, N]`.
- Normalise inputs using TRAIN storms only (mean and std). Save these numbers so `predict.py` can reuse them.
- Make a PyTorch `Dataset` and `DataLoader`.
- Add `tests/ml/test_dataset.py` (check shapes, that no test storm leaks into train).
- **Done when:** tests pass and you print one batch shape.

### B2: Score card (weight 7)
- File: `src/ml/metrics.py`.
- Functions: MAE (in cm), RMSE (in cm), flood/no-flood precision, recall and F1 (flooded = depth > 0.10 m), and an "IoU" for the flooded area. Each one **per lead time** (+1h, +2h, +3h).
- Because floods are rare, also report F1 only on storms that contain a flood, and say clearly which one you mean.
- **Done when:** tests pass on tiny hand-made examples (perfect prediction gives F1 = 1; all-zero prediction gives recall = 0).

### B3: Baselines to beat (weight 10)
- File: `src/ml/baselines.py`.
- Baseline 1, **Persistence**: assume water stays as it is now (use the current fill/depth estimate; say clearly how you did it).
- Baseline 2, **No-graph model**: a model that looks at each node alone (LightGBM or a small MLP) and ignores neighbours.
- Train and score both on the validation storms. Save to `results/baseline_metrics.csv`.
- **Done when:** the table exists and you've explained in simple words which one is better and why.

### B4: Main model, ST-GNN, and training (weight 20)
- File: `src/ml/model.py` and `src/ml/train.py`.
- Use **plain PyTorch**, no extra GNN library, to avoid install trouble. Use the adjacency from `src/common/io.py::build_adjacency`.
- Design (keep it small): per time step, a graph layer mixes each node's features with its neighbours' (`A_hat @ X @ W`), then a GRU reads the 8 steps, then a linear layer outputs the next 12 depth values per node. Hidden size about 32 to 64.
- Because floods are rare, use a loss that doesn't ignore them (for example MSE with a bigger weight on flooded cells). Explain the choice to me.
- Train with the Adam optimizer, early stopping on validation loss, fixed seeds. Print a loss line per epoch.
- Save the best model to `models/best.pt` and settings to `models/best_meta.json`. Save the training curve chart to `results/training_curve.png`.
- **Done when:** the model beats baseline 1 on validation. If it doesn't beat baseline 2, say so honestly and try up to 3 improvements (more history, different loss weight, bigger hidden size), and report each result.

### B5: `predict.py` for Person C (weight 10)
- File: `src/ml/predict.py` with exactly `predict(nodes, edges, recent, model_path="models/best.pt")` from `CONTRACT.md`. It returns the PREDICTION JSON format.
- Cache the loaded model so repeated calls are fast (under 1 second).
- Add a command line test: `python -m src.ml.predict --demo` that picks a test storm, feeds it the last 8 steps and prints the JSON.
- Add `tests/ml/test_predict.py` (JSON has every node, depths >= 0, keys match the contract).
- **Done when:** the demo prints valid JSON. Then tell me to message Person C: "predict() is ready, pull my branch or `main`."

### B6: Evaluation report (weight 8)
- File: `src/ml/evaluate.py`.
- On the TEST storms only: a table of baselines vs ST-GNN at +1h, +2h, +3h (MAE cm, RMSE cm, flood F1). Save `results/final_metrics.csv` and `results/final_metrics.json`.
- Charts in `results/`: (a) predicted vs actual depth scatter, (b) F1 vs lead time for all models, (c) one example storm: predicted vs actual flood map-like picture over time (heatmap), (d) the 10 worst-predicted nodes.
- Write `results/metrics_explained.md`: each number in one simple sentence, plus an honest "limits" paragraph (simulated labels, small area).
- **Done when:** all files exist and I understand each chart (explain them to me).

### B7: Optimizer v1, simple rules (weight 8)
- File: `src/opt/optimize.py`, function `optimize(...)` from `CONTRACT.md`, with `method="rules"`.
- Use `simulate()` from `src/sim/simulator.py` as the "world".
- Rule idea: turn pumps ON early for pump nodes whose fill is rising or that are predicted to flood; close gates that would send water toward nodes already near full; keep outlets draining.
- Compute `flood_cost` with no control (all pumps off, all gates open) and with the rules. Fill the OPTIMIZER JSON. Save the arrays `results/opt_pump_on.npy` and `results/opt_gate_open.npy`.
- **Done when:** it runs on 3 test storms and prints before and after cost, and the after cost is not worse. If it is worse, fix the rules.

### B8: Optimizer v2, search (weight 8)
- Same file, `method="search"`.
- Random search or a simple hill climber: try about 200 to 500 schedules (pump on/off windows per pump node, gate open/close windows per gate), run `simulate()` for each, keep the best. Start from the rules schedule so it is never worse than it.
- Keep the schedule simple (a few switch times per device) so it is explainable and fast. Total runtime per storm under about 60 seconds.
- Create the final before/after chart `results/optimizer_before_after.png` and a table `results/optimizer_results.csv`.
- **Done when:** search beats rules (or equals it), with numbers. Then tell me to message Person C: "optimize() is ready."

### B9: Real data swap (weight 6)
- When Person A shares the real dataset, I put it in `data/<area_name>/`. Run `check_contract`, re-run training (B1, B3, B4), re-run evaluation (B6) and the optimizer (B7, B8) on it.
- Compare the old (fake) and new (real) numbers in simple words. Retune lightly (at most 3 tries).
- Commit the new `models/best.pt`.
- **Done when:** `predict()` works with the new model and Person C can use it.

### B10: Final scenario numbers (weight 5)
- Make 3 scenarios: `light`, `heavy`, `extreme` (rain peaks of about 25, 60 and 110 mm/h, use fixed seeds). Save the rain arrays in `results/scenarios/`. Tell me the exact file names so I can pass them to Person C.
- For each: prediction accuracy, flood cost before and after optimization, number of flooded nodes.
- Save `results/scenario_summary.json` and a pitch-ready table.
- **Done when:** the summary exists and I can read the three headline numbers aloud.

### B11: Scalability test and handoff (weight 5)
- If Person A provides a second area, run the same pipeline on it (retrain or test transfer) and report the numbers. If not, time how the pipeline scales with node count (e.g. 200, 400, 800 fake nodes) and report training and prediction time.
- Write `docs/ml_notes_for_pitch.md`: 10 simple bullets (what the model is, what it learned, numbers, limits, how real data plugs in, what we'd do next) and 8 likely judge questions with honest answers.
- **Done when:** the file exists and I can answer the 8 questions in my own words.

## 8. Git help (give me commands, I run them)

My branch is `person-b/ml-optimizer`. After each step give me:
```
git add -A
git commit -m "B<step>: <short message>"
git push
```
Every 2 hours, or when A or C says something is merged:
```
git fetch origin
git merge origin/main
```
If a merge conflict appears, don't panic: explain the conflict in simple words and fix it for me. My own folders should rarely conflict.

## 9. If I get stuck or the AI forgets

I may start a new chat and paste this file again. When I do, first read `progress/B.md` and the `results/` folder, tell me in simple words what is already done, and continue from the first step that isn't DONE.

## 10. Start now

Begin with **B0**. When B0 is done, print the STATUS block, then continue straight into B1, and keep going step by step as long as nothing needs me. Stop only if you need something from me.
