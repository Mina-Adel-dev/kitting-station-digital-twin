# Kitting Station 1 Digital Twin  
## PLC Controller + SCADA Dashboard + 3D Replay Viewer

A Python-based digital twin for **Station 1 (Kitting Cell)** in a smart factory line.

This project simulates a full kitting workflow with:
- a PLC-like controller,
- four robotic arms,
- sensor/actuator abstraction,
- live SCADA monitoring,
- event logging,
- 3D replay + analytics from simulation logs.

---

## Features

- **Discrete Event Simulation (DES)** using SimPy
- **PLC-centered workflow** for order processing
- **4 robotic arms**:
  - Picking Arm
  - Kitting Arm
  - Mounting Arm
  - Soldering Arm
- **Sensor layer** for safe read-only state access
- **Actuator layer** for controlled PLC commands
- **Live SCADA dashboard** (PyQt5) with KPIs and arm status
- **Structured event logging** to `.jsonl`
- **3D visualization + timeline replay** from log files
- **Analytics dashboard** with charts and performance metrics
- **Data export** to JSON and CSV at simulation completion

---

## Current Scope (from this codebase)

This version models robotic behavior as **2D point-to-point movement + timed operations** with random failures and repairs.

It already includes:
- pick/place/mount/solder operation flow
- utilization tracking
- failure logging + recovery timing
- KPI tracking and reporting

It does **not yet** implement:
- full inverse/forward kinematics,
- collision geometry,
- torque/force limits based on payload weight/girth.

(These can be added in the next development stage.)

---

## Project Structure

```text
done everything/
├─ main.py                 # Main app, simulation engine, SCADA UI, logging/export
├─ station1_plc.py         # PLC controller and station process orchestration
├─ station1_sensors.py     # Sensor abstraction layer
├─ station1_actuators.py   # Actuator abstraction layer
├─ visual.py               # 3D replay viewer + charts + metrics tabs
└─ station_info_panel.py   # Explanatory panel for station logic in UI
