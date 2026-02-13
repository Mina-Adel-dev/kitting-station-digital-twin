<h1 align="center">Kitting Station 1 Digital Twin</h1>
<p align="center">
  PLC Controller • SimPy DES • SCADA Dashboard • 3D Replay Viewer
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="SimPy" src="https://img.shields.io/badge/SimPy-DES-orange">
  <img alt="PyQt5" src="https://img.shields.io/badge/UI-PyQt5-green">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-success">
</p>

---

## Overview

This project is a digital twin for **Station 1 (Kitting Cell)**.

It simulates a PLC-controlled manufacturing workflow with:

- four robotic arms (Picking, Kitting, Mounting, Soldering),
- sensor/actuator abstraction,
- SCADA monitoring,
- structured event logging,
- replay and KPI analytics from logs.

> Current focus: stable process orchestration and replay-driven analysis.

---

## Features

- PLC-style station sequence (`IDLE -> ... -> RESETTING`)
- Discrete event simulation with **SimPy**
- 4-arm coordinated process flow
- Live SCADA dashboard
- JSONL event logging + JSON/CSV export
- 3D replay viewer with timeline controls
- KPI/analytics charts from runtime logs

---

## Screenshots

<p align="center">
  <img alt="SCADA Dashboard Main View" src="https://github.com/user-attachments/assets/10f755db-cd57-4907-be62-ae697d9f0acf" width="90%" />
</p>

<p align="center">
  <img alt="3D Replay Viewer and Analytics" src="https://github.com/user-attachments/assets/d12223f1-ca04-419d-8317-be3071f57e99" width="90%" />
</p>

---

## Demo Video

- [Watch Demo Video (Google Drive)](https://drive.google.com/file/d/1wkmIakzZ-d28HSQj0nSNOWzl9N3eJn9f/view?usp=drive_link)

---

## Project Structure

```text
.
├── main.py                  # Main runtime, simulation manager, SCADA, export
├── station1_plc.py          # PLC controller and process orchestration
├── station1_sensors.py      # Sensor abstraction layer
├── station1_actuators.py    # Actuator abstraction layer
├── visual.py                # 3D replay + charts + metrics
└── station_info_panel.py    # Station info/help panel
