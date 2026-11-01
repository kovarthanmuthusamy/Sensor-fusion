"""Load test for a running detection service: throughput, latency percentiles and load shedding.

    python scripts/load_test.py --url http://localhost:8000 --sample outputs/sample \\
        --concurrency 1 2 4 8 16 --seconds 15 --out docs/load_test.md

`--sample` is a folder with image.png, lidar.npy and calibration.json (see
`scripts/make_sample_request.py` and `scripts/make_test_model.py`). Each concurrency level runs that
many clients, each sending requests back to back, for the given time. Answers of 200 count towards
throughput and latency; 503 (the service shedding load) and other answers are counted separately.
"""

import argparse
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class Result:
    latencies_ms: list[float] = field(default_factory=list)
    shed: int = 0
    errors: int = 0


def client_loop(
    url: str, files: dict[str, bytes], form: dict[str, str], stop_at: float, out: Result
) -> None:
    with httpx.Client(timeout=60) as client:
        while time.perf_counter() < stop_at:
            start = time.perf_counter()
            try:
                response = client.post(
                    f"{url}/detect",
                    files={
                        "image": ("image.png", files["image"], "image/png"),
                        "lidar": ("lidar.npy", files["lidar"], "application/octet-stream"),
                    },
                    data=form,
                )
            except httpx.HTTPError:
                out.errors += 1
                continue
            elapsed = 1000 * (time.perf_counter() - start)
            if response.status_code == 200:
                out.latencies_ms.append(elapsed)
            elif response.status_code == 503:
                out.shed += 1
                time.sleep(float(response.headers.get("retry-after", "1")) / 10)  # a polite client
            else:
                out.errors += 1


def percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def run_level(
    url: str, files: dict[str, bytes], form: dict[str, str], clients: int, seconds: float
) -> dict[str, float]:
    results = [Result() for _ in range(clients)]
    stop_at = time.perf_counter() + seconds
    threads = [
        threading.Thread(target=client_loop, args=(url, files, form, stop_at, r)) for r in results
    ]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started
    latencies = sorted(x for r in results for x in r.latencies_ms)
    if not latencies:
        return {
            "clients": clients,
            "ok": 0,
            "shed": sum(r.shed for r in results),
            "errors": sum(r.errors for r in results),
        }
    return {
        "clients": clients,
        "ok": len(latencies),
        "throughput": len(latencies) / wall,
        "p50": statistics.median(latencies),
        "p95": percentile(latencies, 0.95),
        "p99": percentile(latencies, 0.99),
        "shed": sum(r.shed for r in results),
        "errors": sum(r.errors for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--out", type=Path, default=None, help="also write a markdown table here")
    args = parser.parse_args()

    files = {
        "image": (args.sample / "image.png").read_bytes(),
        "lidar": (args.sample / "lidar.npy").read_bytes(),
    }
    form = {
        "calibration": (args.sample / "calibration.json").read_text(encoding="utf-8"),
        "score_threshold": "0.5",
    }
    version = httpx.get(f"{args.url}/version", timeout=10).json()
    limits = version["limits"]
    for _ in range(3):  # warm-up requests are not measured
        httpx.post(
            f"{args.url}/detect",
            files={"image": files["image"], "lidar": files["lidar"]},
            data=form,
            timeout=60,
        )
    rows = [run_level(args.url, files, form, c, args.seconds) for c in args.concurrency]

    lines = [
        "# Service under load",
        "",
        f"Model `{version['model']}`, {args.seconds:g} s per level, one real frame per request "
        f"(photo + ~35k LiDAR points + calibration), measured {time.strftime('%Y-%m-%d')} on the "
        f"shared lab workstation, client and server on the same machine. Service limits: "
        f"{limits['max_concurrent']} inferences at once, {limits['max_queue']} allowed to wait; "
        "beyond that it answers 503 (load shedding). Latency is for answered (200) requests.",
        "",
        "| clients | answered | requests/s | p50 ms | p95 ms | p99 ms | shed (503) | errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["ok"]:
            lines.append(
                f"| {r['clients']:.0f} | {r['ok']:.0f} | {r['throughput']:.1f} | {r['p50']:.0f} | "
                f"{r['p95']:.0f} | {r['p99']:.0f} | {r['shed']:.0f} | {r['errors']:.0f} |"
            )
        else:
            lines.append(
                f"| {r['clients']:.0f} | 0 | - | - | - | - | {r['shed']:.0f} | {r['errors']:.0f} |"
            )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
