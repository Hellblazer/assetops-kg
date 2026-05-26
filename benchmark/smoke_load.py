"""Smoke-test the IBM loader in isolation. Reports wall time + stats."""
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

from samyama import SamyamaClient  # noqa: E402

from etl.ibm_loader import load_ibm_data  # noqa: E402


def main():
    data_dir = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1
        else "~/projects/Madhulatha-Sandeep/graph_ws/AssetOpsBench"
    )
    client = SamyamaClient.embedded()
    t0 = time.time()
    stats = load_ibm_data(client, data_dir, "smoke")
    elapsed = time.time() - t0
    print(f"\nLoaded in {elapsed:.1f}s")
    print("Stats:")
    for k in sorted(stats):
        print(f"  {k:25s} = {stats[k]}")


if __name__ == "__main__":
    main()
