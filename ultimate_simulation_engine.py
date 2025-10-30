#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

MODE = "OPTIMIZED"  # "BASELINE" or "OPTIMIZED"

# VANETomo knobs (handoff hygiene)
DWELL_DEFAULT = 3
HANDOFF_MARGIN_DEFAULT = 0.05
REROUTE_COOLDOWN = 5
MAX_REROUTES_PER_VEH = 3

# --------- Simple GRNN (Nadaraya–Watson) ----------
class GRNN(tf.keras.Model):
    def __init__(self, sigma=0.2):
        super().__init__()
        self.sigma = float(sigma); self._X = None; self._y = None
    def call(self, X, training=False):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        d2 = tf.reduce_sum(tf.square(tf.expand_dims(X, 1) - self._X), axis=-1)
        w = tf.exp(-d2 / (2.0 * self.sigma ** 2))
        num = tf.reduce_sum(w * tf.transpose(self._y), axis=1)
        den = tf.reduce_sum(w, axis=1) + 1e-9
        return num / den
    def set_training_data(self, X, y):
        self._X = tf.constant(np.asarray(X, np.float32))
        self._y = tf.constant(np.asarray(y, np.float32).reshape(-1, 1))

def _safe(v, default=0.0):
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default

def train_grnn_on_real_data(features_path):
    df = pd.read_csv(features_path)
    need = ['norm_delay', 'norm_thr', 'veh_count', 'lambda_pkts_s', 'utility']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")
    X = df[['norm_delay', 'norm_thr', 'veh_count', 'lambda_pkts_s']].fillna(0.0).values
    y = df['utility'].fillna(0.0).values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    m = GRNN(0.2); m.set_training_data(X_train, y_train)
    return m

# === VANETomo primitives ===
def invert_delay_to_load(theta_ms, dmin_ms, d_const, beta):
    """
    Delay-to-load inversion (guarded).
    """
    theta = max(1e-6, float(theta_ms))
    dmin = max(1e-6, float(dmin_ms))
    d = max(1e-6, float(d_const))
    b = max(1e-6, float(beta))
    denom = (dmin - theta)
    if denom >= -1e-9:
        denom = max(denom, 1e-6)
    inv = 1.0 / denom
    nume = (inv + 1.0)
    deno = max(1e-9, (inv - 1.0))
    ratio = max(1e-9, nume / deno)
    return (1.0 / (d * b)) * np.log(ratio)

def classify_rsus_vanetomo(rsu_frame, dmin_ms, d_const, beta, thmin, thmax):
    # infer per-RSU load from delay & classify
    loads = []
    for _, row in rsu_frame.iterrows():
        theta = _safe(row["AvgDelay_ms"], 10.0)
        L = invert_delay_to_load(theta, dmin_ms, d_const, beta)
        loads.append(L)
    rsu_frame = rsu_frame.copy()
    rsu_frame["L_inferred"] = loads
    # thresholds on inferred loads
    Lvals = np.array([_safe(x, 0.0) for x in loads])
    if len(Lvals) >= 2 and np.isfinite(Lvals).any():
        L_lo = np.quantile(Lvals, thmin)
        L_hi = np.quantile(Lvals, thmax)
    else:
        L_lo, L_hi = float(np.min(Lvals, initial=0.0)), float(np.max(Lvals, initial=1.0))
    def _cls(L):
        if L <= L_lo: return "least_loaded"
        if L >= L_hi: return "overloaded"
        return "normally_loaded"
    rsu_frame["v_class"] = rsu_frame["L_inferred"].apply(_cls)
    return rsu_frame, L_lo, L_hi

def run_simulation():
    print(f"🚀 Starting simulation in {MODE} mode...")

    rounds_df = pd.read_csv("rounds_data.csv").copy()
    rsu_stats = pd.read_csv("rsu_initial_stats.csv").set_index('RSU')
    with open("network_parameters.json") as f:
        N = json.load(f)

    rsu_params = N["rsu_params"]
    avg_pkt = _safe(N.get("global_avg_packet_size", 500.0), 500.0)

    # VANETomo constants
    van = N.get("vanetomo", {})
    dmin_ms = _safe(van.get("dmin_ms", 1.0), 1.0)
    beta = _safe(van.get("beta", 1.5), 1.5)
    d_const = _safe(van.get("d_const", 1.0), 1.0)
    THMIN_q = _safe(van.get("THMIN_q", 0.25), 0.25)
    THMAX_q = _safe(van.get("THMAX_q", 0.75), 0.75)
    DWELL = int(van.get("dwell_rounds", DWELL_DEFAULT))
    HANDOFF_MARGIN = _safe(van.get("handoff_margin", HANDOFF_MARGIN_DEFAULT), 0.05)

    grnn_model = None
    if MODE == "OPTIMIZED":
        grnn_model = train_grnn_on_real_data("rsu_round_features.csv")

    sim_results = []
    veh_reroute_counts = {}
    veh_last_reroute_round = {}
    veh_last_attach = {}         # for VANETomo dwell-time
    veh_last_attach_round = {}

    # dynamic RSU state each round
    dynamic = rsu_stats.copy()
    dynamic['current_lambda_pkts_s'] = dynamic['total_packets']  # seed (per-trace average)
    dynamic['veh_count'] = 0.0

    # per-round loop
    for t_bin in sorted(rounds_df['t_bin'].unique()):
        round_attach = rounds_df[rounds_df['t_bin'] == t_bin].copy()
        reroutes_this_round = 0

        # recompute veh_count per RSU
        rsu_veh_count = round_attach.groupby('rsu')['veh'].count().reindex(dynamic.index, fill_value=0)
        dynamic['veh_count'] = rsu_veh_count.values

        # λ = vehicles * avg per-veh offered load (packets/s)
        lam = []
        for rsu in dynamic.index:
            lam.append(dynamic.loc[rsu, 'veh_count'] * _safe(rsu_params.get(rsu, {}).get('avg_veh_packets_s', 0.0), 0.0))
        dynamic['current_lambda_pkts_s'] = lam

        # update queuing metrics → delay & throughput proxies
        for rsu in dynamic.index:
            lambda_cur = _safe(dynamic.loc[rsu, 'current_lambda_pkts_s'], 0.0)
            mu = _safe(rsu_params.get(rsu, {}).get('mu_pkts_s', 1e-3), 1e-3)
            rho = min(max(lambda_cur / max(1e-6, mu), 0.0), 10.0)
            dynamic.loc[rsu, 'rho'] = rho
            if rho >= 1.0:
                dynamic.loc[rsu, 'AvgDelay_ms'] = 1000.0
                dynamic.loc[rsu, 'AvgThroughput_mbps'] = (mu * avg_pkt * 8.0 / 1_000_000.0) * 0.95
            else:
                service_time = 1.0 / max(1e-6, mu)
                wait_time = (rho / (mu * (1.0 - rho))) if (1.0 - rho) > 1e-6 else service_time
                total_delay_sec = service_time + wait_time
                dynamic.loc[rsu, 'AvgDelay_ms'] = total_delay_sec * 1000.0
                dynamic.loc[rsu, 'AvgThroughput_mbps'] = lambda_cur * avg_pkt * 8.0 / 1_000_000.0

        # ======== VANETomo pass (pre/post-congestion control) ========
        if MODE == "OPTIMIZED":
            v_input = dynamic[["AvgDelay_ms", "AvgThroughput_mbps", "veh_count"]]
            vstate, L_lo, L_hi = classify_rsus_vanetomo(
                v_input, dmin_ms, d_const, beta, THMIN_q, THMAX_q
            )

            # Copy BOTH columns back, index-aligned (fix for KeyError 'L_inferred')
            for col in ["v_class", "L_inferred"]:
                dynamic[col] = vstate[col].reindex(dynamic.index)

            # choose target pools (sort by inferred load ascending → least_loaded first)
            overloaded = dynamic.index[dynamic["v_class"] == "overloaded"]
            if (dynamic["v_class"] == "least_loaded").any():
                least_loaded = list(dynamic[dynamic["v_class"] == "least_loaded"]
                                    .sort_values("L_inferred", ascending=True).index)
            else:
                # Fallback: pick lowest delay if no least_loaded found
                least_loaded = list(dynamic.sort_values("AvgDelay_ms", ascending=True).index[:1])

            # vehicle-level handoff decisions with dwell & margin
            if len(least_loaded) and len(overloaded):
                best_target = least_loaded[0]
                best_util = _safe(1.0 / (1.0 + _safe(dynamic.loc[best_target, "AvgDelay_ms"], 1000.0)), 0.0)

                for idx, row in round_attach.iterrows():
                    veh = row["veh"]; src = row["rsu"]
                    last_rsu = veh_last_attach.get(veh, src)
                    last_r = veh_last_attach_round.get(veh, -999)
                    dwell_ok = (t_bin - last_r) >= DWELL or (last_rsu != src)

                    if (src in set(overloaded)) and dwell_ok:
                        src_util = _safe(1.0 / (1.0 + _safe(dynamic.loc[src, "AvgDelay_ms"], 1000.0)), 0.0)
                        if (best_util - src_util) >= HANDOFF_MARGIN:
                            # switch to VANETomo-chosen target
                            rounds_df.loc[idx, "rsu"] = best_target
                            veh_last_attach[veh] = best_target
                            veh_last_attach_round[veh] = t_bin
                            # update loads immediately
                            veh_load = _safe(rsu_params.get(src, {}).get("avg_veh_packets_s", 0.0), 0.0)
                            dynamic.loc[src, 'current_lambda_pkts_s'] = max(0.0, dynamic.loc[src, 'current_lambda_pkts_s'] - veh_load)
                            dynamic.loc[best_target, 'current_lambda_pkts_s'] += veh_load
                            dynamic.loc[src, 'veh_count'] = max(0, dynamic.loc[src, 'veh_count'] - 1)
                            dynamic.loc[best_target, 'veh_count'] = dynamic.loc[best_target, 'veh_count'] + 1
                            reroutes_this_round += 1

        # re-evaluate delay/throughput after VANETomo moves
        for rsu in dynamic.index:
            lambda_cur = _safe(dynamic.loc[rsu, 'current_lambda_pkts_s'], 0.0)
            mu = _safe(rsu_params.get(rsu, {}).get('mu_pkts_s', 1e-3), 1e-3)
            rho = min(max(lambda_cur / max(1e-6, mu), 0.0), 10.0)
            dynamic.loc[rsu, 'rho'] = rho
            if rho >= 1.0:
                dynamic.loc[rsu, 'AvgDelay_ms'] = 1000.0
                dynamic.loc[rsu, 'AvgThroughput_mbps'] = (mu * avg_pkt * 8.0 / 1_000_000.0) * 0.95
            else:
                service_time = 1.0 / max(1e-6, mu)
                wait_time = (rho / (mu * (1.0 - rho))) if (1.0 - rho) > 1e-6 else service_time
                total_delay_sec = service_time + wait_time
                dynamic.loc[rsu, 'AvgDelay_ms'] = total_delay_sec * 1000.0
                dynamic.loc[rsu, 'AvgThroughput_mbps'] = lambda_cur * avg_pkt * 8.0 / 1_000_000.0

        # ======== GRNN pass (second-stage load balancing) ========
        if MODE == "OPTIMIZED" and grnn_model is not None:
            # round-normalized features for GRNN scoring
            def _minmax(col):
                v = dynamic[col].values.astype(float)
                lo, hi = np.nanmin(v), np.nanmax(v)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
                    return np.full_like(v, 0.5, dtype=float)
                return (v - lo) / (hi - lo)
            norm_delay = _minmax("AvgDelay_ms")
            norm_thr = _minmax("AvgThroughput_mbps")
            features = np.c_[norm_delay, norm_thr,
                             dynamic["veh_count"].values.astype(float),
                             dynamic["current_lambda_pkts_s"].values.astype(float)]
            scores = grnn_model(features).numpy().reshape(-1)
            dynamic["grnn_score"] = scores

            # classify by quantiles
            finite_scores = scores[np.isfinite(scores)]
            if finite_scores.size:
                ql, qh = np.quantile(finite_scores, [0.2, 0.8])
            else:
                ql, qh = 0.0, 1.0
            dynamic["g_class"] = np.where(scores >= qh, "least_loaded",
                                   np.where(scores <= ql, "overloaded", "normally_loaded"))

            overloaded = set(dynamic.index[dynamic["g_class"] == "overloaded"])
            least_loaded = list(dynamic[dynamic["g_class"] == "least_loaded"]
                                .sort_values("grnn_score", ascending=False).index)

            if len(overloaded) and len(least_loaded):
                target = least_loaded[0]
                for idx, row in round_attach.iterrows():
                    veh = row["veh"]; src = rounds_df.loc[idx, "rsu"]  # after VANETomo step
                    if veh not in veh_reroute_counts:
                        veh_reroute_counts[veh] = 0
                    if (veh_reroute_counts[veh] < MAX_REROUTES_PER_VEH and
                        (t_bin - veh_last_reroute_round.get(veh, -999) > REROUTE_COOLDOWN)):
                        if src in overloaded:
                            rounds_df.loc[idx, "rsu"] = target
                            # update loads
                            load_pkts = _safe(rsu_params.get(src, {}).get("avg_veh_packets_s", 0.0), 0.0)
                            dynamic.loc[src, 'current_lambda_pkts_s'] = max(0.0, dynamic.loc[src, 'current_lambda_pkts_s'] - load_pkts)
                            dynamic.loc[target, 'current_lambda_pkts_s'] = dynamic.loc[target, 'current_lambda_pkts_s'] + load_pkts
                            dynamic.loc[src, 'veh_count'] = max(0, dynamic.loc[src, 'veh_count'] - 1)
                            dynamic.loc[target, 'veh_count'] = dynamic.loc[target, 'veh_count'] + 1
                            reroutes_this_round += 1
                            veh_reroute_counts[veh] += 1
                            veh_last_reroute_round[veh] = t_bin

                # refresh after GRNN moves
                for rsu in dynamic.index:
                    lambda_cur = _safe(dynamic.loc[rsu, 'current_lambda_pkts_s'], 0.0)
                    mu = _safe(rsu_params.get(rsu, {}).get('mu_pkts_s', 1e-3), 1e-3)
                    rho = min(max(lambda_cur / max(1e-6, mu), 0.0), 10.0)
                    dynamic.loc[rsu, 'rho'] = rho
                    if rho >= 1.0:
                        dynamic.loc[rsu, 'AvgDelay_ms'] = 1000.0
                        dynamic.loc[rsu, 'AvgThroughput_mbps'] = (mu * avg_pkt * 8.0 / 1_000_000.0) * 0.95
                    else:
                        service_time = 1.0 / max(1e-6, mu)
                        wait_time = (rho / (mu * (1.0 - rho))) if (1.0 - rho) > 1e-6 else service_time
                        total_delay_sec = service_time + wait_time
                        dynamic.loc[rsu, 'AvgDelay_ms'] = total_delay_sec * 1000.0
                        dynamic.loc[rsu, 'AvgThroughput_mbps'] = lambda_cur * avg_pkt * 8.0 / 1_000_000.0

        # aggregate metrics over active RSUs in this round
        active_rsus = rounds_df[rounds_df['t_bin'] == t_bin]['rsu'].unique()
        if len(active_rsus) > 0:
            cur_delay = np.nanmean(dynamic.loc[active_rsus, "AvgDelay_ms"].values.astype(float))
            cur_thr = np.nanmean(dynamic.loc[active_rsus, "AvgThroughput_mbps"].values.astype(float))
        else:
            cur_delay, cur_thr = np.nan, np.nan

        sim_results.append({
            "round": int(t_bin),
            "avg_delay_ms": float(cur_delay),
            "avg_throughput_mbps": float(cur_thr),
            "reroutes": int(reroutes_this_round)
        })

    final_df = pd.DataFrame(sim_results).dropna()
    out = f"simulation_metrics_{MODE.lower()}.csv"
    final_df.to_csv(out, index=False)
    print(f"✅ Simulation complete. Wrote to {out}")

if __name__ == "__main__":
    run_simulation()
