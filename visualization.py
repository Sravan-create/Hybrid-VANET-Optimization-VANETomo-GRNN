#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (prevents macOS blocking)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# === Load CSVs ===
baseline = pd.read_csv("simulation_metrics_baseline.csv")
optimized = pd.read_csv("simulation_metrics_optimized.csv")

# === Prepare x-axis (rounds as vehicle count proxy) ===
x_vals = sorted(set(baseline["round"]).intersection(optimized["round"]))

# === Compute mean per round ===
b_thr = baseline.groupby("round")["avg_throughput_mbps"].mean().reindex(x_vals)
o_thr = optimized.groupby("round")["avg_throughput_mbps"].mean().reindex(x_vals)
b_dly = baseline.groupby("round")["avg_delay_ms"].mean().reindex(x_vals)
o_dly = optimized.groupby("round")["avg_delay_ms"].mean().reindex(x_vals)

# === Plot Throughput ===
plt.figure(figsize=(8,5))
plt.plot(x_vals, b_thr, 'yH--', linewidth=2, markersize=8, label='VANETomo (Baseline)')
plt.plot(x_vals, o_thr, 'ro--', linewidth=2, markersize=8, label='Hybrid (VANETomo + GRNN)')
plt.xlabel('Number of Vehicles (proxy: rounds)', fontsize=12)
plt.ylabel('Average Throughput (Mbps)', fontsize=12)
plt.title('Average Throughput vs Number of Vehicles')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11, frameon=False)
plt.tight_layout()
plt.savefig("throughput_comparison_dotted.png", dpi=300)
plt.close()

# === Plot Delay ===
plt.figure(figsize=(8,5))
plt.plot(x_vals, b_dly, 'yH--', linewidth=2, markersize=8, label='VANETomo (Baseline)')
plt.plot(x_vals, o_dly, 'ro--', linewidth=2, markersize=8, label='Hybrid (VANETomo + GRNN)')
plt.xlabel('Number of Vehicles (proxy: rounds)', fontsize=12)
plt.ylabel('Average Delay (ms)', fontsize=12)
plt.title('Average Delay vs Number of Vehicles')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11, frameon=False)
plt.tight_layout()
plt.savefig("delay_comparison_dotted.png", dpi=300)
plt.close()

print("✅ Saved:")
print("  • throughput_comparison_dotted.png")
print("  • delay_comparison_dotted.png")
