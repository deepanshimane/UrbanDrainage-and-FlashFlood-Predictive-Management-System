# FloodSense Mumbai Dataset (Person A Deliverables)

## Overview
This folder contains the street network, elevation, pump stations, and synthetic/real deluge storm simulation datasets for Mumbai.

Per hackathon data management practices, **large `.npy` arrays, `.tif` elevation rasters, and zip archives are excluded from Git**.

---

## 1. What is in Git (Committed)
- `data/hindmata/nodes.csv` (89 nodes with elevation, drainage area, runoff coefficient, and pump zones)
- `data/hindmata/edges.csv` (140 street/drainage edges with road capacity)
- `data/hindmata/pumps.csv` (3 pumping stations: Hindmata, Gandhi Market, Britannia)
- `data/hindmata/known_spots.csv` & `known_spot_validation.csv` (Hindmata, Dadar TT, Gandhi Market hit-rate validation)
- `data/hindmata/real_storms/` (Open-Meteo extreme rain day CSVs: 2019, 2020, 2021)
- `data/hindmata/dataset/meta.csv` (306 storm metadata records with 70/15/15 train/val/test split)
- `data/hindmata/dataset/config.json` & `scenarios.json` (Dashboard and model configs)

---

## 2. Where the Large Data Arrays Live (.npy)
- Local Path: `data/hindmata/dataset/X.npy` (39.9 MB, shape `[306, 96, 89, 4]`)
- Features: `0 = rain_mm`, `1 = tide_m`, `2 = depth_m` (ground truth target), `3 = pump_on`
- Auxiliary files: `data/hindmata/dataset/gate_open.npy` `[306, 96]`, `pump_sched.npy` `[306, 96, 3]`
- **Local Generation:** If not downloaded from Google Drive, the entire 306-storm dataset can be generated locally in ~1 second by running:
  ```bash
  python gen_dataset.py data/hindmata 300
  ```
- **Google Drive Archive Link:** (Upload `data/hindmata/dataset.zip` to Google Drive and paste link here)
