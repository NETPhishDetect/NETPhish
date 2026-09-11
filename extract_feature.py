import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
from scapy.all import IP, TCP, IPv6, rdpcap
from scapy.layers.tls.all import TLS

CSV_COLUMNS = [
    "size_bidirectional_std",
    "size_bidirectional_mad",
    "size_bidirectional_max",
    "size_bidirectional_min",
    "size_bidirectional_mean",
    "size_direction_pct_10",
    "size_direction_pct_20",
    "size_direction_pct_30",
    "size_direction_pct_40",
    "size_direction_pct_50",
    "size_direction_pct_60",
    "size_direction_pct_70",
    "size_direction_pct_80",
    "size_direction_pct_90",
    "label",
    "size_outcoming_std",
    "size_outcoming_mad",
    "size_outcoming_max",
    "size_outcoming_min",
    "size_outcoming_mean",
    "ip",
    "flow_bytes",
    "flow_num",
    "flow_dur",
    "file_name",
    "size_incoming_std",
    "size_incoming_mad",
    "size_incoming_max",
    "size_incoming_min",
    "size_incoming_mean",
    "time_bidirectional_std",
    "time_bidirectional_mad",
    "time_bidirectional_max",
    "time_bidirectional_min",
    "time_bidirectional_mean",
    "time_direction_pct_10",
    "time_direction_pct_20",
    "time_direction_pct_30",
    "time_direction_pct_40",
    "time_direction_pct_50",
    "time_direction_pct_60",
    "time_direction_pct_70",
    "time_direction_pct_80",
    "time_direction_pct_90",
    "time_incoming_std",
    "time_incoming_mad",
    "time_incoming_max",
    "time_incoming_min",
    "time_incoming_mean",
    "time_outcoming_std",
    "time_outcoming_mad",
    "time_outcoming_max",
    "time_outcoming_min",
    "time_outcoming_mean",
]


def _safe_stats(values):
    if values is None:
        return (None, None, None, None, None)

    values = list(values)
    if len(values) == 0:
        return (None, None, None, None, None)

    arr = np.asarray(values, dtype=float)
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    median = np.median(arr)
    mad = float(np.median(np.abs(arr - median)))
    return (minimum, maximum, mean, std, mad)


def _safe_percentiles(values, ps=(10, 20, 30, 40, 50, 60, 70, 80, 90)):
    result = {}
    if values is None or len(values) == 0:
        for p in ps:
            result[p] = None
        return result

    arr = np.asarray(values, dtype=float)
    for p in ps:
        result[p] = float(np.percentile(arr, p))
    return result


def _get_ip_pair(pkt):
    if IP in pkt:
        return pkt[IP].src, pkt[IP].dst
    if IPv6 in pkt:
        return pkt[IPv6].src, pkt[IPv6].dst
    return None, None


def get_flow_key(pkt, local_ip=None):
    if IP not in pkt:
        return None

    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst

    if TCP not in pkt:
        return None

    sport = pkt[TCP].sport
    dport = pkt[TCP].dport
    proto = 6

    if local_ip is not None:
        src_local = src_ip == local_ip
        dst_local = dst_ip == local_ip
        if not (src_local or dst_local):
            return None
        if src_local and not dst_local:
            return (src_ip, sport, dst_ip, dport, proto)
        if dst_local and not src_local:
            return (dst_ip, dport, src_ip, sport, proto)

    return (src_ip, sport, dst_ip, dport, proto)


def process_pcap(packets, local_ip=None):
    flows = OrderedDict()
    has_clienthello = set()

    for pkt in packets:
        key = get_flow_key(pkt, local_ip=local_ip)
        if key is None:
            continue

        flows.setdefault(key, [])

        if hasattr(pkt, "haslayer") and pkt.haslayer(TLS):
            tls = pkt["TLS"]
            if "TLSClientHello" in tls:
                has_clienthello.add(key)

        flows[key].append(pkt)

    for key in flows:
        flows[key].sort(key=lambda p: float(p.time))

    packet_groups = []
    for key, pkts in flows.items():
        if key not in has_clienthello:
            continue
        if pkts:
            packet_groups.append(sorted(pkts, key=lambda p: float(p.time)))

    packet_groups.sort(key=lambda group: float(group[0].time) if group else 0)
    return packet_groups


def extract_features(pkts, label):
    feat = {}

    tls_pkts = []
    for pkt in pkts:
        if TCP not in pkt:
            continue
        payload = bytes(pkt[TCP].payload)
        if len(payload) > 5:
            tls_pkts.append(pkt)

    tls_pkts.sort(key=lambda p: float(p.time))

    if len(tls_pkts) == 0:
        return {}

    client_ip, server_ip = _get_ip_pair(tls_pkts[0])
    feat["ip"] = server_ip

    t0 = float(tls_pkts[0].time)
    t1 = float(tls_pkts[-1].time)
    feat["flow_dur"] = float(max(0.0, t1 - t0))

    feat["tls_num"] = int(len(tls_pkts))
    feat["flow_num"] = int(len(tls_pkts))

    sizes_bi = [len(pkt[TCP].payload) for pkt in tls_pkts]
    feat["flow_bytes"] = int(np.sum(sizes_bi))

    sizes_in, sizes_out = [], []
    times_in, times_out = [], []
    times_bi = [float(pkt.time) for pkt in tls_pkts]

    for pkt in tls_pkts:
        src, dst = _get_ip_pair(pkt)
        if src is None or dst is None:
            continue

        payload_len = len(pkt[TCP].payload)
        if client_ip is not None and src == client_ip:
            sizes_out.append(payload_len)
            times_out.append(float(pkt.time))
        elif client_ip is not None and dst == client_ip:
            sizes_in.append(payload_len)
            times_in.append(float(pkt.time))
        else:
            sizes_in.append(payload_len)
            times_in.append(float(pkt.time))

    mn, mx, mean, std, mad = _safe_stats(sizes_in)
    feat["size_incoming_min"] = mn
    feat["size_incoming_max"] = mx
    feat["size_incoming_mean"] = mean
    feat["size_incoming_std"] = std
    feat["size_incoming_mad"] = mad

    mn, mx, mean, std, mad = _safe_stats(sizes_out)
    feat["size_outcoming_min"] = mn
    feat["size_outcoming_max"] = mx
    feat["size_outcoming_mean"] = mean
    feat["size_outcoming_std"] = std
    feat["size_outcoming_mad"] = mad

    mn, mx, mean, std, mad = _safe_stats(sizes_bi)
    feat["size_bidirectional_min"] = mn
    feat["size_bidirectional_max"] = mx
    feat["size_bidirectional_mean"] = mean
    feat["size_bidirectional_std"] = std
    feat["size_bidirectional_mad"] = mad

    percentiles = _safe_percentiles(sizes_out)
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        feat[f"size_direction_pct_{p}"] = percentiles[p]

    def _iat(ts):
        ts = list(ts)
        if len(ts) < 2:
            return []
        return list(np.diff(np.asarray(ts, dtype=float)))

    iat_in = _iat(times_in)
    iat_out = _iat(times_out)
    iat_bi = _iat(times_bi)

    mn, mx, mean, std, mad = _safe_stats(iat_in)
    feat["time_incoming_min"] = mn
    feat["time_incoming_max"] = mx
    feat["time_incoming_mean"] = mean
    feat["time_incoming_std"] = std
    feat["time_incoming_mad"] = mad

    mn, mx, mean, std, mad = _safe_stats(iat_out)
    feat["time_outcoming_min"] = mn
    feat["time_outcoming_max"] = mx
    feat["time_outcoming_mean"] = mean
    feat["time_outcoming_std"] = std
    feat["time_outcoming_mad"] = mad

    mn, mx, mean, std, mad = _safe_stats(iat_bi)
    feat["time_bidirectional_min"] = mn
    feat["time_bidirectional_max"] = mx
    feat["time_bidirectional_mean"] = mean
    feat["time_bidirectional_std"] = std
    feat["time_bidirectional_mad"] = mad

    percentiles = _safe_percentiles(iat_out)
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        feat[f"time_direction_pct_{p}"] = percentiles[p]

    feat["label"] = label
    return feat


def save_features_to_csv(all_features, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(CSV_COLUMNS)

        for feat in all_features:
            row = []
            for column in CSV_COLUMNS:
                value = feat.get(column, "")
                if value is None:
                    value = ""
                row.append(value)
            writer.writerow(row)


def find_pcap_files(directory):
    pcap_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith(".pcap"):
                pcap_files.append(os.path.join(root, file_name))
    return sorted(pcap_files)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features from a directory of PCAP files."
    )
    parser.add_argument("input_dir", help="Directory containing PCAP files.")
    parser.add_argument(
        "--label",
        type=int,
        required=True,
        help="Label for the extracted features (Hosting 0, Phishing 1).",
    )
    parser.add_argument("--output", default="flow_features.csv", help="Output CSV file path.")
    parser.add_argument(
        "--local-ip",
        default=None,
        help="Optional local IP used to keep only flows involving this host.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_file = args.output
    label = args.label

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pcap_files = find_pcap_files(input_dir)
    if not pcap_files:
        raise FileNotFoundError(f"No .pcap files found under: {input_dir}")

    all_features = []
    for pcap_path in pcap_files:
        print(f"Processing: {pcap_path}")
        packets = rdpcap(pcap_path)
        packet_groups = process_pcap(packets, local_ip=args.local_ip)
        print(f"Flow count: {len(packet_groups)}")

        for pkts in packet_groups:
            feature = extract_features(pkts, label=label)
            if not feature:
                continue
            feature["file_name"] = os.path.basename(pcap_path)
            all_features.append(feature)

    save_features_to_csv(all_features, output_file)
    print(f"Done. Extracted {len(all_features)} feature rows to {output_file}")


if __name__ == "__main__":
    main()
