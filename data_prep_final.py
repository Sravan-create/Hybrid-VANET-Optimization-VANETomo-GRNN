#!/usr/bin/env python3
import pandas as pd
import json
import numpy as np

NS3_LOG = "ns3__log.csv"
OUT_ROUNDS_DATA = "rounds_data.csv"
OUT_RSU_INITIAL_STATS = "rsu_initial_stats.csv"
OUT_NETWORK_PARAMS = "network_parameters.json"
OUT_RSU_ROUND_FEATURES = "rsu_round_features.csv"  # for GRNN training

def _safe_num(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def prepare_data(path: str):
    df = pd.read_csv(path)
    need = {"recv_wall_ns", "delay_ms", "edge", "veh", "bytes"}
    if not need.issubset(df.columns):
        miss = need - set(df.columns)
        raise ValueError(f"Input CSV is missing columns: {miss}")

    # sanitize
    for c in ["recv_wall_ns", "delay_ms", "bytes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=need, inplace=True)
    df["veh"] = df["veh"].astype(str)
    df["edge"] = df["edge"].astype(str)
    df["t_bin"] = (df["recv_wall_ns"] * 1e-9).astype(int)  # 1s bins
    df = df.sort_values(["t_bin", "veh"]).reset_index(drop=True)

    # (t_bin, veh) -> last RSU seen in that second
    attach = df.groupby(['t_bin', 'veh'])['edge'].last().reset_index()
    attach.rename(columns={'edge': 'rsu'}, inplace=True)
    attach.to_csv(OUT_ROUNDS_DATA, index=False)
    print(f"✅ Wrote per-round vehicle-RSU attachments to {OUT_ROUNDS_DATA}")

    total_seconds = max(1, int(df["t_bin"].nunique()))
    global_avg_pkt_bytes = _safe_num(df["bytes"].mean(), default=500.0)  # bytes/packet (proxy)

    # RSU base stats across whole trace
    thr = df.groupby("edge")["bytes"].sum().reset_index()
    thr["total_packets"] = thr["bytes"] / global_avg_pkt_bytes
    thr["AvgThroughput_mbps"] = (thr["bytes"] * 8.0) / (total_seconds * 1_000_000.0)

    dly = df.groupby("edge")["delay_ms"].mean().reset_index().rename(columns={"delay_ms": "AvgDelay_ms"})
    rsu_stats = pd.merge(dly, thr[["edge", "AvgThroughput_mbps", "total_packets"]], on="edge")
    rsu_stats.rename(columns={"edge": "RSU"}, inplace=True)
    rsu_stats.to_csv(OUT_RSU_INITIAL_STATS, index=False)
    print(f"✅ Wrote initial RSU stats to {OUT_RSU_INITIAL_STATS}")

    # Per-RSU average per-vehicle load proxies
    veh_counts_per_rsu = df.groupby('edge')['veh'].nunique()
    bytes_per_rsu = df.groupby('edge')['bytes'].sum()

    rsu_params = {}
    for rsu in rsu_stats['RSU']:
        num_veh = int(veh_counts_per_rsu.get(rsu, 1))
        total_bytes = _safe_num(bytes_per_rsu.get(rsu, 0.0))
        avg_bytes_per_veh = total_bytes / max(1, num_veh)
        avg_veh_packets_s = (avg_bytes_per_veh / global_avg_pkt_bytes) / total_seconds

        # λ initialization proxies (packets/s)
        total_packets = _safe_num(rsu_stats.loc[rsu_stats['RSU'] == rsu, 'total_packets'].values[0], 0.0)
        lambda_init_pkts_s = total_packets / total_seconds

        # crude service rate proxy from delay (avoid 0)
        D_ms = _safe_num(rsu_stats.loc[rsu_stats['RSU'] == rsu, 'AvgDelay_ms'].values[0], 1.0)
        D_sec = max(1e-3, D_ms / 1000.0)
        mu_pkts_s = lambda_init_pkts_s + (1.0 / D_sec)  # stable if mu > λ

        rsu_params[rsu] = {
            "mu_pkts_s": max(1e-3, mu_pkts_s),
            "avg_veh_packets_s": max(0.0, avg_veh_packets_s)
        }

    # VANETomo constants (delay→load inversion + thresholds)
    dmin_ms = max(0.1, float(df["delay_ms"].quantile(0.02)))  # proxy for d_min
    beta = 1.5     # shape parameter (tunable)
    d_const = 1.0  # scaling (tunable)
    THMIN_q = 0.25
    THMAX_q = 0.75

    network_params = {
        "rsu_params": rsu_params,
        "global_avg_packet_size": global_avg_pkt_bytes,
        "vanetomo": {
            "dmin_ms": dmin_ms,
            "beta": beta,
            "d_const": d_const,
            "THMIN_q": THMIN_q,
            "THMAX_q": THMAX_q,
            "dwell_rounds": 3,        # avoid ping-pong
            "handoff_margin": 0.05     # minimal utility margin to switch
        }
    }
    with open(OUT_NETWORK_PARAMS, 'w') as f:
        json.dump(network_params, f, indent=2)
    print(f"✅ Calculated network parameters and wrote to {OUT_NETWORK_PARAMS}")

    # ========= GRNN training features per round, per RSU =========
    feats = []
    for t in sorted(df['t_bin'].unique()):
        df_t = df[df['t_bin'] == t]
        g = df_t.groupby('edge').agg(
            AvgDelay_ms=('delay_ms', 'mean'),
            total_bytes=('bytes', 'sum'),
            veh_count=('veh', 'nunique')
        ).reset_index()
        g["AvgThroughput_mbps"] = (g["total_bytes"] * 8.0) / 1_000_000.0  # per 1s
        g["lambda_pkts_s"] = g["total_bytes"] / global_avg_pkt_bytes       # per 1s
        g["t_bin"] = t
        g.rename(columns={'edge': 'RSU'}, inplace=True)
        feats.append(g)
    features_df = pd.concat(feats, ignore_index=True)

    # normalize within each round for utility target
    def _per_round_norm(sub):
        for col in ["AvgDelay_ms", "AvgThroughput_mbps"]:
            v = sub[col].values.astype(float)
            lo, hi = np.nanmin(v), np.nanmax(v)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
                if col == "AvgDelay_ms":
                    sub["norm_delay"] = 0.5
                else:
                    sub["norm_thr"] = 0.5
            else:
                if col == "AvgDelay_ms":
                    sub["norm_delay"] = (v - lo) / (hi - lo)
                else:
                    sub["norm_thr"] = (v - lo) / (hi - lo)
        nd = sub.get("norm_delay", pd.Series([0.5]*len(sub)))
        nt = sub.get("norm_thr", pd.Series([0.5]*len(sub)))
        sub["utility"] = 0.6 * (1.0 - nd) + 0.4 * nt
        return sub

    features_df = features_df.groupby("t_bin", group_keys=False).apply(_per_round_norm)
    features_df.to_csv(OUT_RSU_ROUND_FEATURES, index=False)
    print(f"✅ Wrote RSU round features for GRNN training to {OUT_RSU_ROUND_FEATURES}")

if __name__ == "__main__":
    prepare_data(NS3_LOG)
