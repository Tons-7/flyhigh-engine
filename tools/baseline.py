"""Record or check the performance and quality baseline of this machine: bench/baselines/<gpu>.json.

Runs the release CLI (build it first: `cargo build --release`; the wasm size needs tools/build_web.sh) for a fixed set
of texbench configurations on every kernel class and backend the GPU offers, the quality regression on the compare
sets (default and fast render), the parity golden, and notes the wasm size and the engine commit.

    python tools/baseline.py                   # record: write the file for this GPU (commit it after a deliberate change)
    python tools/baseline.py --check           # compare this machine against its committed file; exit 1 on a regression
    python tools/baseline.py --check --quick   # one small configuration, no quality or parity: the structure check in CI

A regression is a pair or mid time more than 3% slower, a quality mean more than 0.05 dB lower or the parity more than
0.1 dB lower than the committed file; the speed goal is "not below these numbers" (bench/README.md).
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLI = ROOT / "target" / "release" / ("flyhigh-dev.exe" if os.name == "nt" else "flyhigh-dev")
SETS = ROOT / "web" / "compare" / "sets"
TOL_TIME = 0.03
TOL_QUALITY_DB = 0.05
TOL_PARITY_DB = 0.1
KEY = ("backend", "class", "w", "h", "iw", "ih", "fast", "linear", "static")
# (texbench arguments, environment, required): an optional configuration is skipped where its backend is missing
TEXBENCH = [
    (["1280", "720"], {}, True),
    (["1920", "1080"], {}, True),
    (["1920", "1080", "--fast", "--linear"], {}, True),
    (["1920", "1080", "--fast", "--linear", "--static"], {}, True),
    (["1280", "720", "--internal", "1280x720", "--fast", "--linear"], {}, True),
    (["1920", "1080"], {"FH_NO_COOP": "1"}, True),
    (["1920", "1080", "--fast", "--linear"], {"FH_NO_COOP": "1"}, True),
    (["1920", "1080"], {"FH_NO_COOP": "1", "FH_PRECISION": "f32"}, True),
    (["1920", "1080", "--fast", "--linear"], {"FH_NO_COOP": "1", "FH_PRECISION": "f32"}, True),
]
if os.name == "nt":  # DX12 compiles through the DXC DLLs beside the CLI (tools/fetch_dxc.sh), else the numbers are FXC's
    TEXBENCH += [
        (["1920", "1080"], {"FH_BACKEND": "dx12"}, False),
        (["1920", "1080", "--fast", "--linear"], {"FH_BACKEND": "dx12"}, False),
    ]
QUICK = [(["640", "360"], {}, True)]


def run(args, env=None, required=True):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run([str(CLI), *args], cwd=ROOT, env=e, capture_output=True, text=True)
    if p.returncode != 0:
        if required:
            sys.exit(f"{' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
        return None
    return p.stdout


def label(r):
    return (f"{r['backend']:<6} {r['class']:<12} {r['w']}x{r['h']} internal {r['iw']}x{r['ih']} "
            f"fast={int(r['fast'])} linear={int(r['linear'])} static={int(r['static'])}")


def key_of(r):
    return tuple(r.get(k, "vulkan" if k == "backend" else None) for k in KEY)


def texbench(configs, runs):
    rows, seen = [], set()
    for args, env, required in configs:
        r = None
        # the autotune picks between candidates a few percent apart, so one process run swings up to 10%: the row is the
        # best of `runs` processes (texbench itself already takes the min of 30 repetitions inside a run)
        for _ in range(runs):
            out = run(["texbench", *args, "--json"], env, required)
            if out is None:
                break
            cur = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])
            if r is None:
                r = cur
            elif timed(cur) and timed(r):
                r["pair_ms"], r["mid_ms"] = min(r["pair_ms"], cur["pair_ms"]), min(r["mid_ms"], cur["mid_ms"])
        if r is None:
            print(f"  texbench {' '.join(args)} {env}: skipped, backend not available")
            continue
        if key_of(r) in seen:  # a GPU without tensor cores or f16 repeats a class
            continue
        seen.add(key_of(r))
        rows.append(r)
        compiler = f" ({r['compiler']})" if r.get("compiler") else ""
        print(f"  {label(r)}{compiler}: pair {r['pair_ms']} ms, mid {r['mid_ms']} ms")
    return rows


def measure(quick, runs):
    out = {"date": datetime.date.today().isoformat(),
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()}
    rows = texbench(QUICK if quick else TEXBENCH, 1 if quick else runs)
    out["gpu"] = rows[0]["gpu"]
    for r in rows:
        r.pop("gpu")
    out["texbench"] = rows
    if quick:
        return out
    if SETS.exists():
        q = {}
        for name, extra in (("default", []), ("fast", ["--fast"])):
            tmp = ROOT / "target" / f"quality_{name}.json"
            run(["quality", str(SETS), "--save", str(tmp), *extra])
            q[name] = {s: v["mean"] for s, v in json.load(open(tmp)).items()}
            print(f"  quality {name}: {q[name]}")
        out["quality_db"] = q
    else:
        print("  compare sets not present, quality skipped")
    fixtures = ROOT / "fixtures" / "v6p"
    if fixtures.exists():
        m = re.search(r"render PSNR = ([\d.]+) dB", run([], {"FH_FIXTURES": str(fixtures)}))
        out["parity_db"] = float(m.group(1)) if m else None
        print(f"  parity {out['parity_db']} dB")
    wasm = ROOT / "web" / "pkg" / "flyhigh_web_bg.wasm"
    if wasm.exists():
        out["wasm_bytes"] = wasm.stat().st_size
    return out


def timed(r):
    return all(isinstance(r.get(k), (int, float)) and r[k] > 0 for k in ("pair_ms", "mid_ms"))


def validate(out):
    for r in out["texbench"]:
        missing = [k for k in (*KEY, "pair_ms", "mid_ms", "tensors_mb") if k not in r]
        if missing:
            sys.exit(f"malformed texbench row, missing {missing}: {r}")
        if not timed(r):  # texbench reports null times on a device without timestamp queries
            print(f"  {label(r)}: no GPU times on this device")


def compare(out, base):
    bad = []
    rows = {key_of(r): r for r in base.get("texbench", [])}
    print(f"  against {base['commit']} ({base['date']}): time within {TOL_TIME:.0%}, quality within {TOL_QUALITY_DB} dB")
    for r in out["texbench"]:
        b = rows.get(key_of(r))
        if b is None or not timed(r) or not timed(b):
            print(f"  {label(r)}: new, no baseline row")
            continue
        d_pair, d_mid = r["pair_ms"] / b["pair_ms"] - 1, r["mid_ms"] / b["mid_ms"] - 1
        verdict = "REGRESSION" if max(d_pair, d_mid) > TOL_TIME else "ok"
        print(f"  {label(r)}: pair {r['pair_ms']:.3f} ms ({d_pair:+.1%}), mid {r['mid_ms']:.3f} ms ({d_mid:+.1%}) {verdict}")
        if verdict != "ok":
            bad.append(f"{label(r)}: pair {d_pair:+.1%}, mid {d_mid:+.1%}")
    for name, cur in out.get("quality_db", {}).items():
        for s, v in cur.items():
            b = base.get("quality_db", {}).get(name, {}).get(s)
            if b is not None and v < b - TOL_QUALITY_DB:
                bad.append(f"quality {name} {s}: {v} dB, baseline {b}")
    p, b = out.get("parity_db"), base.get("parity_db")
    if p is not None and b is not None and p < b - TOL_PARITY_DB:
        bad.append(f"parity {p} dB, baseline {b}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare against the committed file instead of writing it")
    ap.add_argument("--quick", action="store_true", help="one small configuration, no quality or parity (CI structure check)")
    ap.add_argument("--runs", type=int, default=3, help="processes per configuration, the best counts (default 3)")
    a = ap.parse_args()
    if not CLI.exists():
        sys.exit(f"build the CLI first: {CLI}")
    out = measure(a.quick, a.runs)
    validate(out)
    slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"\(.*?\)|vulkan|dx12|metal", "", out["gpu"].lower())).strip("-")
    dest = ROOT / "bench" / "baselines" / f"{slug}.json"
    if not a.check:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2) + "\n", newline="\n")
        print(f"wrote {dest.relative_to(ROOT)}")
        return
    if not dest.exists():
        print(f"no baseline for this GPU ({dest.relative_to(ROOT)}): structure checked only")
        return
    bad = compare(out, json.load(open(dest)))
    if bad:
        sys.exit("regression:\n  " + "\n  ".join(bad))
    print("no regression")


if __name__ == "__main__":
    main()
