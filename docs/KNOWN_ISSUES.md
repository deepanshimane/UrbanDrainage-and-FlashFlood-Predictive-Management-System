# FloodSense: Known Issues & Cross-Team Requests

## 👤 Ask Person A (Data & Physics Simulator)
1. **Time Step & Graph Compatibility for Hindmata:**
   - *Current State:* `data/hindmata` on `origin/Person-A` uses 10-minute time steps ($T=96$) and 89 nodes with a custom `X.npy` format, whereas `models/best.pt` is trained on the master contract format ($DT=15\text{m}$, $T=72$, $N=200$).
   - *Request for Person A:* Generate a standard `data/hindmata` dataset using the master contract schema (`nodes.csv`, `edges.csv`, and individual `.npy` files) so the trained GNN model and optimizer can be benchmarked directly on Hindmata.
2. **Class-based Simulator vs Functional Interface:**
   - *Current State:* `origin/Person-A` refactored `src/sim/simulator.py` into a class `FloodSim`.
   - *Request for Person A:* Ensure backwards compatibility by retaining the standalone `simulate(nodes, edges, rain, river, pump_on, gate_open, dt)` function signature required by Person B's optimizer.

## 👤 Ask Person B (ML Model & Optimizer)
1. **Model Retraining for Area A (Hindmata):**
   - *Current State:* `models/best.pt` is trained exclusively on $N=200$.
   - *Request for Person B:* Once Person A provides contract-compliant Hindmata arrays, execute training script `src/ml/train.py` to produce `models/hindmata_best.pt`.
2. **Genetic Algorithm Option for Optimizer:**
   - *Current State:* Optimizer implements `"rules"` and `"search"`.
   - *Request for Person B:* Confirm whether `"genetic"` option will be included before final code freeze.

## 👤 Person C Internal Tracking (Mine)
1. **Done:** Full API (`src/api/main.py`) with cached routes and latency benchmarks.
2. **Done:** Interactive multi-tab dashboard (`dashboard/app.py`) with map, +1h/+2h/+3h forecasts, optimizer verification, and IoT gauge.
3. **Done:** Automated demo launcher script (`run_demo.ps1`).
4. **Done:** Full documentation suite (demo script, slides outline, QA cheatsheet, pre-demo checklist).

