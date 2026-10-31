# FloodSense Mumbai Dataset (Person A Deliverables)

## Overview
This folder contains the street network, elevation, pump stations, and synthetic/real deluge storm simulation datasets for Mumbai.

Per hackathon data management practices, **large `.npy` arrays, `.tif` elevation rasters, and zip archives are excluded from Git**.

---

## 1. Areas Available
### Area 1 (Primary): Hindmata - Dadar - Parel (`data/hindmata/`)
- `nodes.csv` (89 nodes with elevation, drainage area, runoff coeff 0.85, pump zones)
- `edges.csv` (140 street/drainage edges with road capacity)
- `pumps.csv` (3 stations: Hindmata mini pumping station, Gandhi Market, Britannia)
- `known_spots.csv` & `known_spot_validation.csv` (Hindmata junction floods in 77.1% of heavy storms)
- `dataset/meta.csv` (306 storm metadata records with 70/15/15 train/val/test split)
- `dataset/config.json` & `dataset/scenarios.json`

### Area 2 (Scalability Proof): Kurla - Mithi River (`data/kurla/`)
- `nodes.csv` (91 nodes, elevation 4.35m to 13.19m along Mithi river)
- `edges.csv` (126 street/drainage edges)
- `pumps.csv` (3 stations: Mithi River CST Road tidal gates, Kranti Nagar, Kurla Station West)
- `known_spots.csv` & `known_spot_validation.csv` (Bail Bazar floods in 71.4% of heavy storms)
- `dataset/meta.csv` (106 storm metadata records)
- `dataset/config.json` & `dataset/scenarios.json`

---

## 2. Where Large Data Arrays Live (.npy)
- Hindmata: `data/hindmata/dataset/X.npy` (39.9 MB, shape `[306, 96, 89, 4]`)
- Kurla: `data/kurla/dataset/X.npy` (14.1 MB, shape `[106, 96, 91, 4]`)
- **Local Generation:** To generate large arrays locally on any machine in seconds without downloading:
  ```bash
  python gen_dataset.py data/hindmata 300
  python gen_dataset.py data/kurla 100
  ```
