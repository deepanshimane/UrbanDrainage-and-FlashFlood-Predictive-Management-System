"""IoT sensor stream simulator for FloodSense.
Streams synthetic IoT ultrasonic river/street gauge readings for a given storm and node.
Person C (Backend & Dashboard) can read from stdout or receive POST requests.

Usage:
  python iot_stream.py data/hindmata 93 50 --delay 0.5
  python iot_stream.py data/hindmata 93 50 --endpoint http://localhost:8000/sensor
  python iot_stream.py data/hindmata 93 50 --max-steps 5
"""
import sys, os, time, json, argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="IoT Sensor Stream Simulator")
    parser.add_argument("area", nargs="?", default="data/hindmata", help="Area directory (e.g., data/hindmata)")
    parser.add_argument("storm_id", nargs="?", type=int, default=93, help="Storm index from dataset/meta.csv")
    parser.add_argument("node", nargs="?", type=int, default=50, help="Node ID representing the gauge location (default: 50 - Hindmata)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between 10-minute timesteps")
    parser.add_argument("--endpoint", type=str, default=None, help="Optional HTTP POST endpoint URL (e.g. http://localhost:8000/sensor)")
    parser.add_argument("--max-steps", type=int, default=None, help="Exit after N steps (for testing)")
    args = parser.parse_args()

    x_path = os.path.join(args.area, "dataset", "X.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Dataset not found at {x_path}. Run gen_dataset.py first.")

    X = np.load(x_path, mmap_mode="r")
    S, T, N, F = X.shape

    if args.storm_id >= S:
        raise ValueError(f"storm_id {args.storm_id} out of range (0..{S-1})")
    if args.node >= N:
        raise ValueError(f"node {args.node} out of range (0..{N-1})")

    rng = np.random.default_rng(args.storm_id + args.node)
    post_func = None
    if args.endpoint:
        import requests
        post_func = requests.post

    total_steps = args.max_steps if args.max_steps is not None else T
    print(f"# Starting IoT stream: area={args.area}, storm={args.storm_id}, gauge_node={args.node}, steps={total_steps}", flush=True)

    for t in range(min(T, total_steps)):
        rain_val = float(X[args.storm_id, t, args.node, 0])
        tide_val = float(X[args.storm_id, t, args.node, 1])
        # Sensor measurement: actual depth + slight sensor jitter
        true_depth = float(X[args.storm_id, t, args.node, 2])
        jitter = float(rng.normal(0, 0.003))
        measured_depth = max(0.0, true_depth + jitter)

        msg = {
            "t": t,
            "timestep_min": t * 10,
            "sensor_id": f"gauge_node_{args.node}",
            "water_level_m": round(measured_depth, 4),
            "rain_mm_10min": round(rain_val, 2),
            "tide_m": round(tide_val, 3),
            "status": "FLOOD_ALERT" if measured_depth > 0.10 else "NORMAL"
        }

        print(json.dumps(msg), flush=True)

        if post_func:
            try:
                post_func(args.endpoint, json=msg, timeout=1.0)
            except Exception as e:
                print(f"# Warning: Failed to POST to {args.endpoint}: {e}", file=sys.stderr)

        if args.delay > 0 and t < total_steps - 1:
            time.sleep(args.delay)

if __name__ == "__main__":
    main()

