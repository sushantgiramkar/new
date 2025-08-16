#!/usr/bin/env python3
"""
AI-driven DDoS detection & mitigation (streaming + optional offline CSV).
- Live mode: sniffs packets, builds features per source IP in short windows, uses IsolationForest to flag anomalies.
- Offline mode: reads CSV with per-IP features and outputs detections.

Security note: Mitigation commands are disabled by default (print-only). Review carefully before enabling.
"""

import argparse
import time
import threading
from collections import defaultdict, Counter, deque
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --- Optional: live capture (root privileges typically required) ---
try:
    from scapy.all import sniff, IP, IPv6, TCP, UDP
    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False


# ---------- Feature Engineering ----------

def shannon_entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if n == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / n
        ent -= p * log2(p)
    return ent

def build_feature_row(stats) -> dict:
    """Convert raw window stats for one src IP into model features."""
    pkt_count = stats["pkt_count"]
    byte_count = stats["byte_count"]

    # Ratios and simple robustness against div-by-zero
    syns = stats["tcp_syn"]
    acks = stats["tcp_ack"]
    syn_ratio = syns / max(1, syns + acks)

    udp_count = stats["udp_count"]
    tcp_count = stats["tcp_count"]
    udp_ratio = udp_count / max(1, udp_count + tcp_count)

    dst_port_entropy = shannon_entropy(stats["dst_ports"])
    dst_ip_entropy = shannon_entropy(stats["dst_ips"])

    unique_dst_ports = len(stats["dst_ports"])
    unique_dst_ips = len(stats["dst_ips"])

    mean_pkt_size = byte_count / max(1, pkt_count)

    return dict(
        pkt_count=pkt_count,
        byte_count=byte_count,
        mean_pkt_size=mean_pkt_size,
        udp_ratio=udp_ratio,
        syn_ratio=syn_ratio,
        unique_dst_ports=unique_dst_ports,
        unique_dst_ips=unique_dst_ips,
        dst_port_entropy=dst_port_entropy,
        dst_ip_entropy=dst_ip_entropy,
        tcp_count=tcp_count,
        udp_count=udp_count,
    )

FEATURE_COLUMNS = [
    "pkt_count","byte_count","mean_pkt_size","udp_ratio","syn_ratio",
    "unique_dst_ports","unique_dst_ips","dst_port_entropy","dst_ip_entropy",
    "tcp_count","udp_count"
]


# ---------- Streaming Aggregator ----------

class WindowAggregator:
    def __init__(self, window_seconds=5):
        self.window_seconds = window_seconds
        self.lock = threading.Lock()
        self.reset_window()

    def reset_window(self):
        self.start_time = time.time()
        self.by_src = defaultdict(lambda: dict(
            pkt_count=0,
            byte_count=0,
            tcp_count=0,
            udp_count=0,
            tcp_syn=0,
            tcp_ack=0,
            dst_ports=Counter(),
            dst_ips=Counter(),
        ))

    def add_packet(self, src, dst, proto, length, tcp_flags=None, dport=None):
        with self.lock:
            s = self.by_src[src]
            s["pkt_count"] += 1
            s["byte_count"] += int(length)
            s["dst_ips"][dst] += 1
            if dport is not None:
                s["dst_ports"][int(dport)] += 1
            if proto == "TCP":
                s["tcp_count"] += 1
                if tcp_flags:
                    if tcp_flags.get("S", 0) == 1 and tcp_flags.get("A", 0) == 0:
                        s["tcp_syn"] += 1
                    if tcp_flags.get("A", 0) == 1:
                        s["tcp_ack"] += 1
            elif proto == "UDP":
                s["udp_count"] += 1

    def should_emit(self):
        return (time.time() - self.start_time) >= self.window_seconds

    def emit_features(self):
        with self.lock:
            rows = []
            src_ips = []
            for src, stats in self.by_src.items():
                rows.append(build_feature_row(stats))
                src_ips.append(src)
            df = pd.DataFrame(rows, index=src_ips)
            self.reset_window()
        return df


# ---------- Anomaly Model ----------

class AnomalyDetector:
    """
    IsolationForest-based detector with rolling baseline.
    """
    def __init__(self, contamination=0.01, baseline_windows=12, random_state=42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,  # expected outlier fraction per window
            random_state=random_state,
            warm_start=False
        )
        self.ready = False
        self.baseline_windows = baseline_windows
        self._buffer = deque(maxlen=baseline_windows)

    def partial_fit(self, df_features: pd.DataFrame):
        # Collect baseline data until buffer fills, then fit once and refit modestly thereafter.
        if df_features.empty:
            return
        self._buffer.append(df_features[FEATURE_COLUMNS].values)
        if not self.ready and len(self._buffer) >= self.baseline_windows:
            X = np.vstack(list(self._buffer))
            self.model.fit(X)
            self.ready = True
        elif self.ready:
            # Refresh model with recent baseline (prevents concept drift)
            X = np.vstack(list(self._buffer))
            self.model.fit(X)

    def score(self, df_features: pd.DataFrame) -> pd.Series:
        if df_features.empty:
            return pd.Series(dtype=float)
        X = df_features[FEATURE_COLUMNS].values
        if not self.ready:
            # Not enough baseline yet; use a heuristic score (higher = worse)
            # Heuristic: emphasize high packet count and low dst entropy (spray behavior)
            heur = (
                (df_features["pkt_count"] / (df_features["dst_ip_entropy"] + 1e-3)) +
                (df_features["pkt_count"] / (df_features["dst_port_entropy"] + 1e-3)) +
                (100 * df_features["syn_ratio"])
            )
            return (heur - heur.min()) / max(1e-9, (heur.max() - heur.min()))
        # IsolationForest returns higher scores for normal points with score_samples
        iso_scores = self.model.score_samples(X)
        # Convert to anomaly score: lower iso_scores => more anomalous
        anom = -iso_scores
        # Normalize per-window for readability
        return (anom - anom.min()) / max(1e-9, (anom.max() - anom.min()))

    def predict_labels(self, scores: pd.Series, threshold: float = 0.85) -> pd.Series:
        # Label as malicious if normalized anomaly score ≥ threshold
        return (scores >= threshold).astype(int)


# ---------- Mitigation (safe-by-default) ----------

class Mitigator:
    def __init__(self, enable_system_block=False, block_seconds=300, blocklist_path="blocklist.txt"):
        self.enable_system_block = enable_system_block
        self.block_seconds = block_seconds
        self.blocklist_path = Path(blocklist_path)
        self.blocked_until = {}  # ip -> epoch timestamp

    def block(self, ip: str, reason: str, score: float):
        now = time.time()
        until = now + self.block_seconds
        # Update local memory
        self.blocked_until[ip] = until
        # Persist suggested block
        line = f"{int(now)},{ip},{int(until)},{reason},{score:.3f}\n"
        self.blocklist_path.write_text(self.blocklist_path.read_text() + line if self.blocklist_path.exists() else line)

        # Print action (safe default)
        print(f"[MITIGATION] Would block {ip} for {self.block_seconds}s (score={score:.2f}) reason={reason}")

        # OPTIONAL (disabled): run iptables/nftables. Review carefully before enabling.
        if self.enable_system_block:
            import subprocess
            try:
                # Example (iptables): drop all traffic from IP (IPv4). Adjust for your environment.
                subprocess.run(["sudo", "iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"], check=True)
                print(f"[SYSTEM] iptables DROP inserted for {ip}")
            except Exception as e:
                print(f"[SYSTEM] Failed to insert iptables rule for {ip}: {e}")

    def cleanup(self):
        """Placeholder to remove expired blocks if you manage them programmatically."""
        now = time.time()
        expired = [ip for ip, until in self.blocked_until.items() if until <= now]
        for ip in expired:
            del self.blocked_until[ip]
            # If you inserted firewall rules, remove them here.


# ---------- Live Mode ----------

def run_live(args):
    if not SCAPY_AVAILABLE:
        raise RuntimeError("scapy is not available. Install scapy or use --offline.")

    agg = WindowAggregator(window_seconds=args.window)
    det = AnomalyDetector(contamination=args.contamination, baseline_windows=args.baseline)
    mit = Mitigator(enable_system_block=args.block, block_seconds=args.block_seconds, blocklist_path=args.blocklist)

    def on_packet(pkt):
        try:
            # Support IPv4 and IPv6
            ip_layer = pkt.getlayer(IP) or pkt.getlayer(IPv6)
            if ip_layer is None:
                return
            src = ip_layer.src
            dst = ip_layer.dst
            length = int(len(pkt))
            dport = None
            proto = "OTHER"
            flags_dict = None

            if pkt.haslayer(TCP):
                t = pkt.getlayer(TCP)
                proto = "TCP"
                dport = t.dport
                flags_dict = {
                    "F": int(t.flags & 0x01 != 0),
                    "S": int(t.flags & 0x02 != 0),
                    "R": int(t.flags & 0x04 != 0),
                    "P": int(t.flags & 0x08 != 0),
                    "A": int(t.flags & 0x10 != 0),
                    "U": int(t.flags & 0x20 != 0),
                    "E": int(t.flags & 0x40 != 0),
                    "C": int(t.flags & 0x80 != 0),
                }
            elif pkt.haslayer(UDP):
                u = pkt.getlayer(UDP)
                proto = "UDP"
                dport = u.dport

            agg.add_packet(src, dst, proto, length, tcp_flags=flags_dict, dport=dport)
        except Exception:
            pass  # keep the sniffer robust

    print(f"[INFO] Sniffing on interface '{args.iface}' with BPF='{args.bpf}' ... (Ctrl+C to stop)")
    sniffer = threading.Thread(target=lambda: sniff(
        iface=args.iface if args.iface else None,
        prn=on_packet,
        store=False,
        filter=args.bpf if args.bpf else None,
    ), daemon=True)
    sniffer.start()

    try:
        while True:
            time.sleep(0.2)
            if agg.should_emit():
                df = agg.emit_features()
                if df.empty:
                    continue

                # Update baseline and score
                det.partial_fit(df)
                scores = det.score(df)
                labels = det.predict_labels(scores, threshold=args.threshold)

                # Report
                window_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                df_out = df.copy()
                df_out["anomaly_score"] = scores
                df_out["is_malicious"] = labels
                df_out.index.name = "src_ip"

                # Sort by score descending for visibility
                df_out = df_out.sort_values("anomaly_score", ascending=False)

                print(f"\n=== Window @ {window_time} (n={len(df_out)}) ===")
                print(df_out[["anomaly_score","pkt_count","byte_count","syn_ratio","udp_ratio","unique_dst_ips","unique_dst_ports"]].head(20).round(3))

                # Mitigation
                for ip, row in df_out.iterrows():
                    if row["is_malicious"] == 1:
                        reason = "DDoS-like anomaly"
                        mit.block(ip, reason, float(row["anomaly_score"]))

                mit.cleanup()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")


# ---------- Offline Mode (CSV of per-IP features) ----------

def run_offline(args):
    """
    CSV should have columns matching FEATURE_COLUMNS plus an optional 'src_ip' column.
    If multiple windows per IP exist, they are evaluated row-wise.
    """
    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if "src_ip" in df.columns:
        df = df.set_index("src_ip")

    # Minimal validation
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    det = AnomalyDetector(contamination=args.contamination, baseline_windows=args.baseline)
    # Build a rolling baseline using the first chunk, then score the rest
    # Here we simply use all rows to fit (unsupervised), then score (for demo).
    det.partial_fit(df[FEATURE_COLUMNS])
    scores = det.score(df)
    labels = det.predict_labels(scores, threshold=args.threshold)
    out = df.copy()
    out["anomaly_score"] = scores
    out["is_malicious"] = labels
    out.index.name = "src_ip"
    out = out.sort_values("anomaly_score", ascending=False)

    print(out[["anomaly_score","pkt_count","byte_count","syn_ratio","udp_ratio","unique_dst_ips","unique_dst_ports"]].head(50).round(3))

    if args.save:
        out_path = Path(args.save)
        out.to_csv(out_path)
        print(f"[INFO] Wrote scored results to {out_path.resolve()}")


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="AI in DDoS attack mitigation (live & offline)")
    sub = p.add_subparsers(dest="mode", required=True)

    # Live
    pl = sub.add_parser("live", help="Sniff packets and detect anomalies in real time")
    pl.add_argument("--iface", default=None, help="Network interface (default: scapy chooses)")
    pl.add_argument("--bpf", default="ip or ip6", help="BPF filter (default: 'ip or ip6')")
    pl.add_argument("--window", type=int, default=5, help="Aggregation window in seconds")
    pl.add_argument("--threshold", type=float, default=0.85, help="Anomaly threshold (0..1)")
    pl.add_argument("--contamination", type=float, default=0.01, help="Expected attack fraction per window")
    pl.add_argument("--baseline", type=int, default=12, help="Windows to build baseline before full model kicks in")
    pl.add_argument("--block", action="store_true", help="Enable system firewall block (OFF by default)")
    pl.add_argument("--block-seconds", type=int, default=300, help="Block duration if --block is set")
    pl.add_argument("--blocklist", default="blocklist.txt", help="Path to append suggested blocks")
    pl.set_defaults(func=run_live)

    # Offline
    pf = sub.add_parser("offline", help="Score a CSV of per-IP features")
    pf.add_argument("--csv", required=True, help="Path to CSV file")
    pf.add_argument("--threshold", type=float, default=0.85, help="Anomaly threshold (0..1)")
    pf.add_argument("--contamination", type=float, default=0.01, help="Expected attack fraction")
    pf.add_argument("--baseline", type=int, default=12, help="Baseline windows before model is 'ready'")
    pf.add_argument("--save", default="", help="Optional: path to write scored CSV")
    pf.set_defaults(func=run_offline)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
