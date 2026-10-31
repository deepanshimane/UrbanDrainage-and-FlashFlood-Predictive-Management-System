# FloodSense: Pre-Demo Readiness Checklist & Emergency Contingencies

## 📋 15-Minute Pre-Demo Checklist

- [ ] **Laptops Charged:** Both primary presentation laptop and backup charged to 100%.
- [ ] **Python Environment Ready:** Verified Python 3.10+ with `torch`, `fastapi`, `streamlit`, `pydeck`.
- [ ] **Dataset Verified:** `python -m src.common.check_contract data/sample` outputs `OK: 60 storms, 72 steps, 200 nodes`.
- [ ] **Model Weights Present:** `models/best.pt` (238 KB) and `models/best_meta.json` present.
- [ ] **Services Running:**
  - FastAPI running on `http://localhost:8000` (test via `http://localhost:8000/health`).
  - Streamlit running on `http://localhost:8501`.
- [ ] **Browser Tabs Pre-Loaded:**
  - Tab 1: Dashboard UI (`http://localhost:8501`).
  - Tab 2: FastAPI Swagger Docs (`http://localhost:8000/docs`).
  - Tab 3: Presentation Slides / Pitch Deck.
- [ ] **Cache Pre-Warmed:** Open optimizer tab once before presentation to ensure responses are instantaneous (<50ms).
- [ ] **Backup Video Ready:** Demo walkthrough video stored locally on desktop in case live browser freezes.

---

## 🚨 Emergency Contingency Plan ("If X breaks, do Y")

| Issue | Immediate Action |
|---|---|
| **FastAPI fails to launch or crashes** | Switch dashboard to **"Direct Import Mode"** in the sidebar. The UI continues running directly with zero interruption. |
| **Pydeck WebGL map doesn't render** | Switch browser hardware acceleration ON, or toggle to 2D Topo scatter mode in the dashboard settings. |
| **Presentation venue Wi-Fi dies** | FloodSense runs **100% locally on localhost** with zero internet dependencies. Do not worry about Wi-Fi. |
| **Optimizer search takes too long** | The dashboard utilizes pre-cached optimizer runs (`results/opt_*.npy`), ensuring results render in <35 ms. |
| **Streamlit server becomes unresponsive** | Run `taskkill /F /IM python.exe` in terminal and relaunch with `.\run_demo.ps1`. Takes under 5 seconds. |
| **Judge asks to test an unseen storm** | Select "Extreme cloudburst" from the sidebar and advance slider to step 26 to show resilience under 110 mm/h deluge. |
