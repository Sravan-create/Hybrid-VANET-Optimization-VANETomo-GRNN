#!/usr/bin/env python3
import pandas as pd

try:
    baseline_df = pd.read_csv("simulation_metrics_baseline.csv")
    optimized_df = pd.read_csv("simulation_metrics_optimized.csv")

    print("\n--- FINAL SIMULATION RESULTS ---")

    # Baseline
    avg_delay_base = float(baseline_df['avg_delay_ms'].mean())
    avg_thr_base = float(baseline_df['avg_throughput_mbps'].mean())
    total_reroutes_base = int(baseline_df['reroutes'].sum())
    print("\n🐌 BASELINE (Replay):")
    print(f"  - Overall Average Delay:      {avg_delay_base:.2f} ms")
    print(f"  - Overall Average Throughput: {avg_thr_base:.4f} Mbps")
    print(f"  - Total Reroutes:             {total_reroutes_base}")

    # Optimized (VANETomo + GRNN)
    avg_delay_opt = float(optimized_df['avg_delay_ms'].mean())
    avg_thr_opt = float(optimized_df['avg_throughput_mbps'].mean())
    total_reroutes_opt = int(optimized_df['reroutes'].sum())
    print("\n🚀 OPTIMIZED (VANETomo + GRNN):")
    print(f"  - Overall Average Delay:      {avg_delay_opt:.2f} ms")
    print(f"  - Overall Average Throughput: {avg_thr_opt:.4f} Mbps")
    print(f"  - Total Reroutes:             {total_reroutes_opt}")

    # Improvements
    delay_reduction = ((avg_delay_base - avg_delay_opt) / avg_delay_base * 100.0) if avg_delay_base > 0 else 0.0
    thr_increase    = ((avg_thr_opt - avg_thr_base) / avg_thr_base * 100.0) if avg_thr_base  > 0 else 0.0
    print("\n--- QUANTIFIABLE IMPROVEMENT ---")
    print(f"✅ Delay Reduction:     {delay_reduction:.2f}%")
    print(f"✅ Throughput Increase: {thr_increase:.2f}%")
    print("----------------------------------\n")

except FileNotFoundError:
    print("Error: Run baseline and optimized first.")
