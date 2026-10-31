# 01 - Person A Plan: Data and Simulator (Mumbai FloodSense)

You are the foundation of the team. Person B (model + optimizer) and Person C (dashboard + demo) both wait for your files. **Working and crude beats perfect and late.**

---

## 1. What you must deliver

| # | Deliverable | Who uses it | Deadline (hour of hackathon) |
|---|---|---|---|
| 1 | `nodes.csv`, `edges.csv`, `pumps.csv` for the main area | B, C | Hour 3 |
| 2 | `simulator.py` (fast, tested) | B (optimizer), me | Hour 4 |
| 3 | Small dataset (50 storms) | B | **Hour 5** |
| 4 | Full dataset (300+ storms, with real storms) + `scenarios.json` | B, C | Hour 8 |
| 5 | Known-flood-spot check ("does our simulator flood Hindmata?") | Pitch | Hour 10 |
| 6 | IoT sensor stream script | C | Hour 10 |
| 7 | Second area (Kurla-Mithi) built with the same scripts | Pitch (scalability) | Hour 12 |
| 8 | README data section + `ASSUMPTIONS.md` | Judges | Hour 13 |

Code and format details are in `02_PERSON_A_IMPLEMENTATION.md` and `03_DATA_CONTRACT.md`.

---

## 2. Area choice

**Main area: Hindmata - Dadar - Parel** (about 3 sq km, radius 1 km around Hindmata junction).
**Second area (scalability proof): Kurla - Mithi River.**

Why this works for the story (all public facts):
- Hindmata is a well-known chronic flood spot. Reports list Hindmata, Sion, King's Circle, Kurla and parts of Dadar as low-lying flood-prone places.
- BMC already has pumping stations (Brimstowad project after 2005) and **mini pumping stations at Hindmata and Gandhi Market**. So "optimize the pumps" is a real problem, not a toy.
- The Mithi river area has tidal floodgates (and 27 more were planned). That matches the problem statement's "sluice gates".
- Judges know Mumbai floods. No need to explain why it matters.

### The risk: flat terrain
Mumbai is flat. Some areas are only about 1 m above sea level, which is about 1.5 m below the high-tide level. So a normal "water flows downhill" model gives boring results.

**How we handle it (this is actually our best story):**
1. In Mumbai, floods are not mainly caused by slope. They come from **rain + drain capacity + high tide together**. The old drains were built to handle only about 25 mm of rain per hour.
2. Many outfalls are blocked or reduced when the tide is high. A BMC statement says only 6 of 186 outfalls are above the high-tide level.
3. So our simulator has **tide as an input** and **tidal gates**. Rain at low tide = small problem. Same rain at high tide = flood. This makes the model interesting even on flat land.
4. Water moves by **water level difference** (ground height + water depth), not just ground slope. So it still spreads on flat land.
5. The DEM (height data) has errors as big as Mumbai's whole height range. We use it only as a rough guide for "which places are lower", and we **check against known flood spots**.

---

## 3. Our world model (simple version)

Every 10 minutes, for each node (junction):

1. **Rain** falls. Runoff = rain x node area x runoff coefficient (0.85, because the area is mostly built-up).
2. **Local drains** remove water at about 25 mm/hour. When the **tide is high** and the gate is closed, drains work at only 10% (low-lying nodes are hit harder).
3. If the gate is **open at high tide**, sea water comes back in (bad).
4. **Pumps** remove water from the nodes in their zone (each node belongs to its nearest pump).
5. **Spreading:** water moves between neighbours based on the water level difference, limited by road/drain capacity.
6. **Flood depth** at a node = water volume / node area. We call it "flooded" if it is more than 10 cm.

Gate rule (automatic baseline): open at low tide, closed at high tide. The optimizer (Person B) can try better gate and pump schedules.

---

## 4. Assumptions (write these into `ASSUMPTIONS.md`)

| Item | Value | Why | Can tune? |
|---|---|---|---|
| Time step | 10 min | Flash floods are fast | No (agreed with team) |
| Episode length | 96 steps = 16 h | Rain + one full tide cycle | No |
| Runoff coefficient | 0.85 | Dense built-up area | Yes |
| Drain design rate | 25 mm/h | Reported design of old Mumbai drains | Yes |
| Drain factor, gate closed | 0.1 | Mostly blocked | Yes |
| Tide | sine wave, period 12.42 h, amplitude 1.5-2.2 m (relative to mean sea level) | Mumbai has two tides a day | Yes |
| Gate level | 1.0 m | Tide above this blocks outfalls (assumption) | Yes |
| Pump capacity | 3 m3/s per station (zone-based) | Real big pumps are about 6 m3/s each; our stations serve a small area | Yes |
| Road capacity | 0.5 to 3 m3/s by road type | Bigger road = bigger drain | Yes |
| Flood threshold | 10 cm | Standard "street flooded" level | No |
| Drainage network | Road network used as stand-in | Real sewer maps are not public | Say it openly |

**Say clearly in the pitch:** the simulator creates the "ground truth", the AI learns from it, and it is tested on storms it has never seen. When real BMC data is available, it plugs in through the same file format.

---

## 5. Data sources

| Need | Source | Notes |
|---|---|---|
| Streets / graph | OpenStreetMap via `osmnx` | Free, any area |
| Height | Copernicus GLO-30 DEM (or FABDEM, which removes buildings; or SRTM) | 30 m resolution; noisy in cities |
| Rain history | Open-Meteo archive API (hourly, free, no login) | Based on reanalysis, so it **underestimates extreme rain**. Say so. |
| Tide | Sine wave in the simulator. Real tide tables (Mumbai port) can be used to check timing. | Mention high tide above 4.5 m + heavy rain is the known dangerous mix |
| Pump locations | Google Maps (right-click, copy coordinates) | Hindmata mini pumping station, Gandhi Market mini pumping station, Britannia pumping station (check which fall inside your radius) |
| Known flood spots | News + BMC lists | Used only to check the simulator |

---

## 6. Your timeline

| Hour | Task | Done when |
|---|---|---|
| 0-1 | With team: agree on `03_DATA_CONTRACT.md` | Everyone says OK |
| 1-2 | Set up Python env, download DEM, list pump coordinates, run `build_area.py` | You have a graph with 200-600 nodes |
| 2-3 | Check graph plot and elevation, fix node count, write `ASSUMPTIONS.md` | `nodes.csv`, `edges.csv`, `pumps.csv` exist |
| 3-4 | Add `simulator.py`, run sanity tests | "ALL SANITY TESTS PASSED" |
| 4-5 | `gen_dataset.py` with 50 storms, send to B | B confirms the shapes load |
| 5-6 | Tune `k`, pump capacity, edge capacity until flooding looks believable | Heavy storm at high tide floods Hindmata area |
| 6-8 | Download 2-3 real heavy-rain days, generate the full dataset (300+) | `dataset/` folder complete |
| 8-10 | Known-spot check, IoT stream script | Hit-rate table done |
| 10-12 | Integration with B and C, fix bugs | End-to-end run works |
| 12-13 | Second area (Kurla-Mithi) with the same scripts | Second `nodes.csv` and dataset |
| 13 | **Code freeze** | No new features |
| 13-15 | README data section, help with demo and slides | Done |

---

## 7. How to know your data is "believable" (validation)

1. **Sanity tests** (in code): zero rain gives zero flood, more rain gives more flood, pumps ON is better than OFF, closed gates at high tide are worse than automatic gates, no negative or NaN water.
2. **Tide story test:** same storm, tide peak at the rain peak vs. tide low. The first must flood more. This is your best demo line.
3. **Known-spot test:** Hindmata, Gandhi Market, King's Circle (if inside your radius) should flood **more often than the average node** in heavy storms.
4. **Look at the map:** low-lying nodes near the pump zones should be the wettest. If flooding is random noise, your DEM is bad (see troubleshooting in `02_`).

---

## 8. Risks and fixes

| Risk | Fix |
|---|---|
| DEM is too flat or noisy | Use FABDEM or smooth more. As a fallback, add a rule: nodes near known low spots get a few cm lower. Say so. |
| Too many nodes (over 600) | Raise `TOL` (merge distance) or reduce `DIST` |
| Pumps have no effect | Pumps pull from their whole zone. Check `pump_zone` is set and `cap_m3s` is big enough. |
| Everything floods or nothing floods | Tune `drain_mm_hr`, `k`, and rain totals. Aim for: light rain = no flood, heavy = some, cloudburst + high tide = a lot. |
| You are late at hour 5 | Hand over what you have. B can use the simple version. Formats must stay the same. |
| Changing file formats later | Never do this without telling B and C first |
| Hackathon rule: "all work created during the hackathon" | Use the reference code as a guide and write and adapt it yourself during the event. Public libraries and datasets are allowed. If unsure, ask the organizers. |

---

## 9. Mumbai facts for the pitch (with sources)

- Old drains built to handle about 25 mm of rain per hour (DNA India, "Rs 10,000 crore spent so far...").
- Only 6 of 186 outfalls are above high-tide level; 45 are below sea level (Free Press Journal, "Pumping stations for a flood-free city this year").
- Some areas lie about 1 m above sea level, about 1.5 m below high tide; the dangerous mix is high tide above 4.5 m plus very heavy rain (Citizen Matters, "Understanding BMC's flood mitigation measures").
- Hindmata, Sion, King's Circle, Kurla and parts of Dadar are low-lying flood-prone spots (Business Today, July 2026).
- Eight Brimstowad pumping stations were proposed after 2005; mini pumping stations at Gandhi Market and Hindmata (Free Press Journal, 2022).
- Tidal gates close at high tide and open at low tide; each large pump can discharge about 6,000 litres per second (Wilo India).
- 27 floodgates planned on the Mithi river from Kurla (CST Road bridge) to Mahim Causeway (India Infra Hub).

Check these again before the event. News and project status can change.

---

## 10. Definition of done

- [ ] `nodes.csv`, `edges.csv`, `pumps.csv` follow the contract
- [ ] `python test_sanity.py` prints ALL SANITY TESTS PASSED
- [ ] Dataset has 300+ storms, 70/15/15 split by storm, includes 2-3 real storms
- [ ] `scenarios.json` has light, heavy, cloudburst
- [ ] Known-spot table made
- [ ] Second area works with the same scripts
- [ ] `ASSUMPTIONS.md` and README data section written
- [ ] B and C confirmed they can load everything
