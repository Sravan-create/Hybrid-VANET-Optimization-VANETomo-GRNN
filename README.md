# Hybrid VANET Optimization: VANETomo + GRNN

A lightweight hybrid optimization pipeline for vehicular networks:

- **Data prep** builds per-round vehicle↔RSU attachments and per-RSU stats from a flat log.
- **Baseline** simulation **replays** those attachments (no learning, no rerouting).
- **Optimized** simulation runs an embedded **VANETomo-style handoff** (load inference from delay + THMIN/THMAX) **then** a **GRNN** to further rebalance load.
- **Compare & visualize** average delay / throughput improvements with dotted-line plots.

---

## What you get

- `data_prep.py` → parses `ns3__log.csv`, writes:
  - `rounds_data.csv`, `rsu_initial_stats.csv`, `network_parameters.json`, `rsu_round_features.csv`
- `ultimate_simulation_engine.py` → runs **BASELINE** (replay) or **OPTIMIZED** (VANETomo + GRNN)
- `compare_files.py` → prints mean metrics and % deltas
- `visualization.py` → saves dotted-line PNGs for throughput/delay comparisons

**Expected input log columns:**  
`recv_wall_ns, delay_ms, edge, veh, bytes`

---

## Quickstart

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

### Prepare data

```bash
python data_prep_final.py
```

### Run baseline (replay only)

Open `ultimate_simulation_engine.py` and set:

```python
MODE = "BASELINE"
```

Then run:

```bash
python ultimate_simulation_engine.py
```

### Run optimized (hybrid: VANETomo + GRNN)

In `ultimate_simulation_engine.py` set:

```python
MODE = "OPTIMIZED"
```

Then run:

```bash
python ultimate_simulation_engine.py
```

### Compare results

```bash
python compare_files.py
```

### Make dotted-line charts (PNG)

```bash
python visualization.py
```

**Outputs**

* `throughput_comparison_dotted.png`
* `delay_comparison_dotted.png`

---

## How it works (very short)

* **Baseline:** Replays per-second attachments and computes per-RSU metrics with simple M/M/1-style proxies; no reroutes.
* **VANETomo pass (inside Optimized):** Infers load from delay, classifies RSUs via quantile thresholds (THMIN/THMAX), and hands off vehicles from overloaded → least-loaded with dwell/hysteresis.
* **GRNN pass (inside Optimized):** Trains on your `rsu_round_features.csv` (delay/throughput/load signals) and performs a second pass of targeted rerouting.

---

## Tips & troubleshooting

* On macOS, if Matplotlib shows a GUI window or blocks execution, the provided `visualization.py` already uses a headless backend and **saves** PNGs instead of showing them.
* If your input log isn’t from VANETomo, that’s fine — the **baseline** is still a replay; the **optimized** run is the hybrid optimizer on top of your data.
* Percent improvements can look large when baselines are very small; report **both absolute and relative** deltas for clarity.

---

## License

MIT (or your preference).

````

---

### requirements.txt
```txt
numpy>=1.24
pandas>=1.5
scikit-learn>=1.3
matplotlib>=3.7
tensorflow-cpu>=2.12,<3.0
````

*(If you have GPU/CUDA, you can replace `tensorflow-cpu` with `tensorflow`.)*
