# FloodSense: Hydrologic & System Engineering Assumptions

This document outlines all engineering constants, hydrologic physics simplifications, and municipal data assumptions used across the FloodSense pipeline (Problem Statement 11, AI Manthan 2.0).

---

## 1. Urban Drainage & Hydrology Assumptions

| Parameter / Item | Assumed Value | Physical & Empirical Justification | Tunable? |
|---|---|---|---|
| **Simulation Timestep ($\Delta t$)** | **10 minutes** ($600\text{ s}$) | Flash flood ponding and pump response in dense urban areas occur within minutes. Matches sensor sampling intervals. | Fixed across team contract |
| **Episode Length ($T$)** | **96 steps = 16 hours** | Captures pre-storm buildup, deluge peak, and at least one complete semi-diurnal tidal cycle ($12.42\text{ h}$). | Fixed |
| **Runoff Coefficient ($c$)** | **0.85** | Dadar, Hindmata, and Parel are predominantly impervious (paved roads, high-density residential/commercial concrete). | Yes ($0.75 - 0.90$) |
| **Drain Design Capacity** | **25 mm/hour** | Sourced from public BRIMSTOWAD reports confirming legacy stormwater drains were designed for $25\text{ mm/h}$ rainfall intensity. | Yes |
| **Tidal Inundation Period** | **12.42 hours** (semi-diurnal) | Mumbai Arabian Sea tides follow a semi-diurnal cycle with two high tides and two low tides daily. Modeled as a harmonic sinusoidal series: $A \sin(2\pi t / 12.42 + \phi)$. | Yes |
| **Tidal Amplitude ($A$)** | **1.5 m to 2.2 m** above MSL | Mumbai astronomical tides typically fluctuate between $+1.0\text{ m}$ (neap) and $+4.8\text{ m}$ CD (approx $+2.2\text{ m}$ above MSL during spring tides). | Yes |
| **Outfall Sluice Gate Level** | **1.0 m above MSL** ($3.51\text{ m}$ CD) | Design assumption based on low-elevation coastal flap inverts (not an official BMC standard). When tide exceeds $1.0\text{ m}$ MSL ($3.51\text{ m}$ CD), gravity flow is choked by sea backwater. | Yes |
| **Closed Gate Outfall Factor** | **0.10 (10% capacity)** | When tidal floodgates are closed to prevent tidal ingress, drain gravity flow is obstructed and drops to residual percolation levels (~10%). | Yes |
| **Open Gate High Tide Backflow** | **10 mm/h backflow rate** | If sluice gates remain stuck open during high tide, seawater surges inward into low-lying catchment zones. | Yes |
| **Flood Inundation Threshold** | **0.10 m (10 cm)** | Standard municipal waterlogging benchmark where curb overflow occurs and vehicular traffic is impaired. | Fixed |
| **Ponding Storage Area** | **2% to 5% of catchment area** | Land surface area reserved for street depressions and gutter storage before uncontrolled surface spreading occurs. | Yes |

---

## 2. Pumping & Actuation Assumptions

| Parameter | Value | Details |
|---|---|---|
| **Hindmata Mini Pumping Station** | **3.0 m³/s** ($3,000\text{ L/s}$) | Underground holding tanks at Pramod Mahajan Kala Park discharging toward arterial nallahs. |
| **Gandhi Market Mini Pumping Station** | **3.0 m³/s** | Submersible pumps dedicated to King's Circle / Gandhi Market depression. |
| **Britannia Stormwater Pumping Station** | **6.0 m³/s** | Regional BRIMSTOWAD outfall pumping station at Sewri / Reay Road discharging into Mumbai Harbour. |
| **Kurla: CST Road Tidal Outfall Gates** | **6.0 m³/s** | Gravity sluice gates and high-capacity pumps on Mithi River bridge. |
| **Kurla: Kranti Nagar Pumping Station** | **3.0 m³/s** | Riverbank pumping station for low-lying Kranti Nagar settlement. |
| **Kurla: Kurla Station West Pumping Station** | **3.0 m³/s** | Dewatering station for Kurla railway subway and market depression. |
| **Pump Zone Allocation** | Nearest-Neighbor Voronoi | Street nodes are serviced by their nearest operational pumping station. Pump drawdown removes water proportionally from available node ponding storage. |
| **Actuation Granularity** | 10-minute intervals | Pumps and sluice gates can be scheduled dynamically per 10-minute horizon. |

---

## 3. Data Source Transparency & Honesty Policy

In full compliance with competition honesty standards:

1. **System Classification:**
   - FloodSense is a **prototype decision-support system that recommends and simulates proactive actions**, not an active municipal SCADA controller.

2. **What is Real:**
   - **Street Network Topology:** Extracted directly from OpenStreetMap (`osmnx`, drive network, simplified intersection clusters).
   - **Topographic Terrain:** Sourced from Copernicus GLO-30 Digital Elevation Model (smoothed over node neighborhood to eliminate urban radar speckle).
   - **Drain Capacity (25 mm/h):** The 25 mm/h capacity is cited from public BRIMSTOWAD reports.
   - **Tide Datum Reference:** Mumbai Port Trust Chart Datum is 2.51 m below Mean Sea Level (MSL = +2.51 m CD, Survey of India datum).

3. **Engineering Assumptions & Simulated Components:**
   - **Tidal Lockout Level (1.0 m MSL / 3.51 m CD):** The 1.0 m MSL outfall lockout threshold is our own engineering design assumption (based on coastal flap invert levels), not an official BMC municipal standard.
   - **Monsoon Storm Scenarios:** The interactive scenarios represent synthetic rainfall hyetographs totaling 43.5 mm (Light), 104.5 mm (Heavy), and 191.6 mm (Extreme) accumulation with harmonic tides, calibrated to typical Mumbai monsoon intensities.
   - **Simulated Inundation:** Runoff accumulation, hydraulic head spreading, and flood depth labels are generated by our high-performance vectorized physical simulator (`simulator.py`).
   - **Extreme Inundation Stress Test (~190 cm):** Depths exceeding 100 cm represent a synthetic stress-test worst case (simultaneous 191.6 mm cloudburst + spring high tide lockout) designed to test hydraulic limits, not an average seasonal forecast.
   - **Dual Baselines & Real Numbers:** We evaluate dispatch schedules against both **Baseline 1: Passive Drainage** (pumps OFF, gates OPEN) and **Baseline 2: Current Municipal Practice** (automatic gates + reactive float-switch pumps). On unseen test storms, proactive dispatch reduces flood volume exposure by an average of **9.4% over current practice** and **10.9% over passive drainage**.

