"""The Python binding against the built C library: an in-between through the Session and the Stream, and a Pacer
driven by synthetic ticks. Run from the repository root after `cargo build --release -p flyhigh-capi`:

    FLYHIGH_LIB=target/release/flyhigh.dll python bindings/python/test_flyhigh.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flyhigh  # noqa: E402

W, H = 256, 192
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")


def frame(dx: float) -> bytearray:
    v = bytearray(W * H * 4)
    for y in range(H):
        w = y * 0.05
        row = y * W * 4
        for x in range(W):
            u = (x - dx) * 0.07
            o = row + x * 4
            v[o] = int(128 + 100 * math.sin(u))
            v[o + 1] = int(128 + 100 * math.cos(u * 0.7 + w))
            v[o + 2] = int(128 + 90 * math.sin(u + w))
            v[o + 3] = 255
    return v


def mae(a, b) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def check_inbetween(got, label):
    f0, f1, mid = frame(0.0), frame(8.0), frame(4.0)
    to_mid, to_f0, to_f1 = mae(got, mid), mae(got, f0), mae(got, f1)
    print(f"{label}: MAE to the true middle {to_mid:.2f}, to f0 {to_f0:.2f}, to f1 {to_f1:.2f}")
    assert to_mid < to_f0 and to_mid < to_f1 and to_mid < 6.0, "the in-between lies between its neighbours"


def main():
    print("library:", flyhigh.library_path())
    with flyhigh.Engine() as engine, flyhigh.Model(MODELS) as model:
        print("engine:", engine.info())
        print("model:", model.info())
        with flyhigh.Session(engine, model, W, H) as s:
            a, b, out = s.add_input(), s.add_input(), s.add_output()
            s.write(a, frame(0.0))
            s.write(b, frame(8.0))
            s.pair(a, b)
            s.mid(0.5, out)
            check_inbetween(s.read(out), "session")
            r = s.report()
            assert (r["width"], r["height"], r["inputs"], r["outputs"]) == (W, H, 2, 1), r
            print("session timed: pair", s.pair_timed(a, b), "ms, mid", s.mid_timed(0.5, out), "ms")
            try:
                s.mid(0.5, 7)
            except flyhigh.FlyHighError as e:
                print("expected error:", e)
            else:
                raise AssertionError("a bad slot must raise")
        with flyhigh.Stream(engine, model, W, H) as st:
            s0 = st.next_input()
            st.write(s0, frame(0.0))
            assert not st.push(s0, 0.0).ready
            s1 = st.next_input([s0])
            st.write(s1, frame(2.0))
            c = st.classify(s0, s1)
            print("classify:", c)
            assert c is not None and c.kind == "motion"
            p = st.push(s1, 41.7)
            assert p.ready and p.base_slot == s0 and p.end_pts == 41.7, p
            assert not st.generate(0.0, 0), "the source frame is enough at t 0"
            assert st.generate(0.5, 0)
            assert not st.warp(0, 0, 42.0, output=True), "no camera warp before any mouse input"
            st.mouse(42.0, 5.0, 0.0)
            st.set_mouse_captured(True)
            assert not st.warp(0, 0, 43.0, output=True), "nor before the calibration holds"
            assert st.report()["camera_warp"]["on"] is False
            print("stream:", st.report()["counters"])
        print("engine report:", engine.report())
    # the Pacer without a GPU: 120 Hz ticks, a 24 fps source, every request generated at once
    with flyhigh.Pacer("auto", min_present_interval_ms=0) as pacer:
        t, next_src, n, presents, drops = 0.0, 5.0, 0, 0, 0
        while t < 6000.0:
            if next_src <= t:
                flush = pacer.source_frame(t, n % 8, n * 1000.0 / 24.0, "motion", 10.0)
                for r in flush:
                    pacer.generated(t, r.id, 0)
                n += 1
                next_src += 1000.0 / 24.0
            tick = pacer.tick(t)
            presents += tick.present is not None
            drops += tick.dropped
            for r in tick.generate:
                pacer.generated(t, r.id, r.id % 8)
            t += 1000.0 / 120.0
        rep = pacer.report()
        print(f"pacer: {presents} presents in 6 s, factor {rep['factor']}, penalty {rep['penalty']}, drops {drops}")
        assert presents > 400 and rep["penalty"] == 0 and drops == 0, rep
    print("ok")


if __name__ == "__main__":
    main()
