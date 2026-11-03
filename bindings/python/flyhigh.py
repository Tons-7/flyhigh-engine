"""FlyHigh Engine for Python: ctypes over the C ABI (crates/flyhigh-capi, include/flyhigh.h).

The engine's layers as classes: Engine (the device), Model, Session (two frames in, one frame at phase t out), Stream
(frames with timestamps in, intervals out) and Pacer (what to present on this tick, what to generate). Frames are RGBA8
bytes (width * height * 4); any object with the buffer protocol works (bytes, bytearray, numpy arrays).

    import flyhigh
    engine = flyhigh.Engine()                         # the device; Engine(precision="f32") forces the f32 path
    model = flyhigh.Model("models")                   # graph.json, flyhighfg.json, flyhighfg.bin
    s = flyhigh.Session(engine, model, 1920, 1080)    # the plans for one size; options as keyword arguments
    a, b, out = s.add_input(), s.add_input(), s.add_output()
    s.write(a, frame0); s.write(b, frame1)
    s.pair(a, b); s.mid(0.5, out)
    rgba = s.read(out)                                # bytes, width * height * 4

The library is found through FLYHIGH_LIB (a path), next to this file, or by name on the loader's path
(flyhigh.dll / libflyhigh.so / libflyhigh.dylib). Every failed call raises FlyHighError with the engine's message.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import (POINTER, Structure, byref, c_char, c_char_p, c_double, c_float, c_int, c_int32, c_size_t, c_uint8,
                    c_uint32, c_uint64, c_void_p, create_string_buffer)
from typing import Optional, Sequence

__all__ = ["FlyHighError", "Engine", "Model", "Session", "Stream", "Pacer", "Classification", "Push", "Request",
           "Tick", "Present", "library_path"]


class FlyHighError(RuntimeError):
    """A failed engine call, with the message of fh_last_error()."""


# ---- the C structs, field for field as in flyhigh.h ----

class EngineOptions(Structure):
    _fields_ = [("backend", c_uint32), ("tensor_cores", c_uint32), ("autotune", c_uint32), ("adapter", c_char_p),
                ("precision", c_uint32), ("tune_cache", c_char_p), ("compact_memory", c_uint32),
                ("dxc_path", c_char_p)]


class SessionOptions(Structure):
    _fields_ = [("width", c_uint32), ("height", c_uint32), ("internal_w", c_uint32), ("internal_h", c_uint32),
                ("k", c_uint32), ("repair", c_uint32), ("render", c_uint32), ("keep_static", c_uint32),
                ("static_frames", c_uint32), ("static_tolerance", c_uint32), ("fast_render", c_uint32),
                ("linear_t", c_uint32), ("keep_maps", c_uint32), ("protection_mask", c_uint32), ("tune", c_char_p)]


class StreamOptions(Structure):
    _fields_ = [("mode", c_uint32), ("cadence", c_uint32), ("scene_cut", c_double), ("held_frames", c_uint32),
                ("held_mean", c_double), ("held_max", c_double), ("inputs", c_uint32), ("outputs", c_uint32)]


class _Classification(Structure):
    _fields_ = [("kind", c_uint32), ("mean", c_float), ("max", c_float), ("black", c_uint32), ("motion", c_double)]


class _Push(Structure):
    _fields_ = [("ready", c_uint32), ("base_slot", c_int32), ("stats_pending", c_uint32), ("start_pts", c_double),
                ("end_pts", c_double)]


class PacerOptions(Structure):
    _fields_ = [("mode", c_uint32), ("factor", c_uint32), ("target_fps", c_double), ("max_factor", c_uint32),
                ("fps_limit", c_double), ("lookahead", c_uint32), ("limit_to_display", c_uint32),
                ("fill_display", c_uint32), ("display_headroom", c_double), ("compute_budget", c_double),
                ("delay_floor_ms", c_double), ("delay_max_ms", c_double), ("lateness_intervals", c_double),
                ("queue_capacity", c_uint32), ("pressure_tau_ms", c_double), ("penalty_up", c_double),
                ("penalty_burst", c_double), ("penalty_min_gap_ms", c_double), ("penalty_max", c_uint32),
                ("recovery_after_ms", c_double), ("stall_window_ms", c_double), ("stall_rate_limit", c_double),
                ("settle_ms", c_double), ("motion_adaptive", c_uint32), ("min_present_interval_ms", c_double),
                ("playback_rate", c_double), ("mid_cost_seed_ms", c_double), ("pair_cost_seed_ms", c_double),
                ("gpu_timestamps", c_uint32), ("low_latency", c_uint32)]


class _SourceFrame(Structure):
    _fields_ = [("id", c_uint32), ("pts_ms", c_double), ("kind", c_uint32), ("has_motion", c_uint32),
                ("motion", c_double), ("skipped", c_uint32)]


class _Request(Structure):
    _fields_ = [("id", c_uint64), ("previous", c_uint32), ("current", c_uint32), ("t", c_float), ("due_ms", c_double),
                ("retime", c_uint32), ("probe", c_uint32)]


class _Tick(Structure):
    _fields_ = [("has_present", c_uint32), ("frame_kind", c_uint32), ("frame_id", c_uint32), ("due_ms", c_double),
                ("late_ms", c_double), ("mid", c_uint32), ("fallback", c_uint32), ("factor", c_uint32),
                ("dropped", c_uint32), ("requests", c_uint32)]


# ---- the library ----

def library_path() -> str:
    """Where the C library is taken from: FLYHIGH_LIB, a file next to this module, else the platform's default name."""
    env = os.environ.get("FLYHIGH_LIB")
    if env:
        return env
    name = "flyhigh.dll" if sys.platform == "win32" else "libflyhigh.dylib" if sys.platform == "darwin" else "libflyhigh.so"
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    return local if os.path.exists(local) else name


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(library_path())
    sig = lib.__getattr__

    def f(name, restype, *argtypes):
        fn = sig(name)
        fn.restype = restype
        fn.argtypes = list(argtypes)
        return fn

    f("fh_last_error", c_char_p)
    f("fh_engine_options_default", EngineOptions)
    f("fh_engine_create", c_void_p, POINTER(EngineOptions))
    f("fh_engine_destroy", None, c_void_p)
    f("fh_engine_info", c_int, c_void_p, c_char_p, c_size_t)
    f("fh_engine_report_json", c_int, c_void_p, c_char_p, c_size_t)
    f("fh_model_load", c_void_p, c_char_p)
    f("fh_model_from_bytes", c_void_p, POINTER(c_uint8), c_size_t, POINTER(c_uint8), c_size_t, POINTER(c_uint8), c_size_t)
    f("fh_model_destroy", None, c_void_p)
    f("fh_model_info", c_int, c_void_p, c_char_p, c_size_t)
    f("fh_session_options_default", SessionOptions, c_uint32, c_uint32)
    f("fh_session_create", c_void_p, c_void_p, c_void_p, POINTER(SessionOptions))
    f("fh_session_destroy", None, c_void_p)
    f("fh_session_add_input", c_int, c_void_p)
    f("fh_session_add_output", c_int, c_void_p)
    f("fh_session_write_rgba", c_int, c_void_p, c_uint32, POINTER(c_uint8), c_size_t)
    f("fh_session_write_mask", c_int, c_void_p, POINTER(c_uint8), c_size_t)
    f("fh_session_read_rgba", c_int, c_void_p, c_uint32, POINTER(c_uint8), c_size_t)
    f("fh_session_pair", c_int, c_void_p, c_uint32, c_uint32)
    f("fh_session_mid", c_int, c_void_p, c_float, c_uint32)
    f("fh_session_generate", c_int, c_void_p, c_uint32, c_uint32, c_float, c_uint32)
    f("fh_session_pair_timed", c_int, c_void_p, c_uint32, c_uint32, POINTER(c_double))
    f("fh_session_mid_timed", c_int, c_void_p, c_float, c_uint32, POINTER(c_double))
    f("fh_session_warm_up", c_int, c_void_p)
    f("fh_session_report_json", c_int, c_void_p, c_char_p, c_size_t)
    f("fh_stream_options_default", StreamOptions)
    f("fh_stream_create", c_void_p, c_void_p, c_void_p, POINTER(SessionOptions), POINTER(StreamOptions))
    f("fh_stream_destroy", None, c_void_p)
    f("fh_stream_add_input", c_int, c_void_p, c_void_p)
    f("fh_stream_add_output", c_int, c_void_p, c_void_p)
    f("fh_stream_next_input", c_int, c_void_p, POINTER(c_uint32), c_size_t)
    f("fh_stream_write_rgba", c_int, c_void_p, c_void_p, c_uint32, POINTER(c_uint8), c_size_t)
    f("fh_stream_write_mask", c_int, c_void_p, c_void_p, POINTER(c_uint8), c_size_t)
    f("fh_stream_classify", c_int, c_void_p, c_void_p, c_uint32, c_uint32, POINTER(_Classification))
    f("fh_stream_push", c_int, c_void_p, c_void_p, c_uint32, c_double, POINTER(_Push))
    f("fh_stream_resolve_stats", c_int, c_void_p, c_void_p)
    f("fh_stream_real_replaced", c_int, c_void_p)
    f("fh_stream_generate", c_int, c_void_p, c_void_p, c_float, c_uint32)
    f("fh_stream_read_rgba", c_int, c_void_p, c_void_p, c_uint32, POINTER(c_uint8), c_size_t)
    f("fh_stream_warm_up", c_int, c_void_p, c_void_p)
    f("fh_stream_reset", c_int, c_void_p)
    f("fh_stream_mouse", c_int, c_void_p, c_void_p, c_double, c_double, c_double)
    f("fh_stream_set_mouse_captured", c_int, c_void_p, c_uint32)
    f("fh_stream_warp", c_int, c_void_p, c_void_p, c_uint32, c_uint32, c_uint32, c_double)
    f("fh_stream_report_json", c_int, c_void_p, c_char_p, c_size_t)
    f("fh_pacer_options_default", PacerOptions)
    f("fh_pacer_create", c_void_p, POINTER(PacerOptions))
    f("fh_pacer_destroy", None, c_void_p)
    f("fh_pacer_set_options", c_int, c_void_p, c_double, POINTER(PacerOptions))
    f("fh_pacer_tick", c_int, c_void_p, c_double, POINTER(_Tick))
    f("fh_pacer_source_frame", c_int, c_void_p, c_double, POINTER(_SourceFrame))
    f("fh_pacer_requests", c_int, c_void_p, POINTER(_Request), c_size_t)
    f("fh_pacer_generated", c_int, c_void_p, c_double, c_uint64, c_uint32)
    f("fh_pacer_skipped", c_int, c_void_p, c_uint64)
    f("fh_pacer_cost_sample", c_int, c_void_p, c_double, c_uint32, c_double)
    f("fh_pacer_host_stall", c_int, c_void_p, c_double, c_double)
    f("fh_pacer_mode_change", c_int, c_void_p, c_double)
    f("fh_pacer_reset", c_int, c_void_p, c_double)
    f("fh_pacer_report_json", c_int, c_void_p, c_char_p, c_size_t)
    f("fh_pacer_record", c_int, c_void_p, c_char_p)
    f("fh_pacer_trace_json", c_int, c_void_p, c_char_p, c_size_t)
    _lib = lib
    return lib


def _error() -> FlyHighError:
    msg = _load().fh_last_error()
    return FlyHighError(msg.decode("utf-8", "replace") if msg else "unknown error")


def _check(rc: int) -> int:
    if rc < 0:
        raise _error()
    return rc


def _json(fn, handle, *args) -> dict:
    cap = 1 << 16
    while True:
        buf = create_string_buffer(cap)
        n = fn(handle, *args, buf, cap)
        if n >= 0:
            return json.loads(buf.raw[:n].decode("utf-8"))
        msg = _load().fh_last_error()
        if msg and b"buffer too small" in msg and cap < (1 << 26):
            cap *= 4
            continue
        raise _error()


def _bytes(data) -> tuple:
    """A buffer as (pointer, length) without a copy where the object supports the buffer protocol."""
    view = memoryview(data).cast("B") if not isinstance(data, (bytes, bytearray)) else memoryview(data)
    if not view.c_contiguous:
        view = memoryview(bytes(view))
    n = view.nbytes
    arr = (c_uint8 * n).from_buffer(view) if not view.readonly else (c_uint8 * n).from_buffer_copy(view)
    return arr, n


_BACKENDS = {"auto": 0, "vulkan": 1, "dx12": 2, "metal": 3, "gl": 4}
_PRECISION = {"auto": 0, "f16": 1, "f32": 2}
_RENDER = {"model": 0, "bilinear": 1, "bicubic": 2}
_MODE = {"realtime": 0, "video": 1}
_KIND = {0: "motion", 1: "cut", 2: "hold"}
_KIND_CODE = {v: k for k, v in _KIND.items()}
_STAGE = {"pair": 0, "mid": 1, "present": 2}


class Engine:
    """The device. Engine(backend="auto"|"vulkan"|"dx12"|"metal", precision="auto"|"f16"|"f32", tensor_cores=True,
    autotune=True, adapter=None, tune_cache=None, compact_memory=True, dxc_path=None). dxc_path (Windows, DX12) is where
    dxcompiler.dll and dxil.dll live when they are not beside flyhigh.dll. Destroy sessions and streams before it (or
    use the objects as context managers)."""

    def __init__(self, backend: str = "auto", precision: str = "auto", tensor_cores: bool = True, autotune: bool = True,
                 adapter: Optional[str] = None, tune_cache: Optional[str] = None, compact_memory: bool = True,
                 dxc_path: Optional[str] = None):
        lib = _load()
        o = lib.fh_engine_options_default()
        o.backend = _BACKENDS[backend]
        o.precision = _PRECISION[precision]
        o.tensor_cores = int(tensor_cores)
        o.autotune = int(autotune)
        o.compact_memory = int(compact_memory)
        self._adapter = adapter.encode() if adapter else None
        self._tune_cache = tune_cache.encode() if tune_cache else None
        self._dxc_path = dxc_path.encode() if dxc_path else None
        o.adapter = self._adapter
        o.tune_cache = self._tune_cache
        o.dxc_path = self._dxc_path
        self._h = lib.fh_engine_create(byref(o))
        if not self._h:
            raise _error()

    def info(self) -> dict:
        return _json(_load().fh_engine_info, self._h)

    def report(self) -> dict:
        return _json(_load().fh_engine_report_json, self._h)

    def close(self):
        if getattr(self, "_h", None):
            _load().fh_engine_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        self.close()


class Model:
    """Weights and graph: Model("models") from a directory, or Model.from_bytes(graph, manifest, weights)."""

    def __init__(self, directory: Optional[str] = None, _handle=None):
        lib = _load()
        if _handle is not None:
            self._h = _handle
        else:
            if directory is None:
                raise ValueError("a directory, or Model.from_bytes")
            self._h = lib.fh_model_load(directory.encode())
        if not self._h:
            raise _error()

    @classmethod
    def from_bytes(cls, graph_json: bytes, manifest_json: bytes, weights: bytes) -> "Model":
        g, gn = _bytes(graph_json)
        m, mn = _bytes(manifest_json)
        w, wn = _bytes(weights)
        h = _load().fh_model_from_bytes(g, gn, m, mn, w, wn)
        return cls(_handle=h)

    def info(self) -> dict:
        return _json(_load().fh_model_info, self._h)

    def close(self):
        if getattr(self, "_h", None):
            _load().fh_model_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        self.close()


def _session_options(width: int, height: int, internal=None, k: int = 1, repair: int = 1, render: str = "model",
                     static_protection=None, fast_render: bool = False, linear_t: bool = False, keep_maps: bool = False,
                     protection_mask: bool = False, tune: Optional[str] = None) -> SessionOptions:
    o = _load().fh_session_options_default(width, height)
    if internal == "frame":
        o.internal_w, o.internal_h = width, height
    elif internal not in (None, "auto"):
        o.internal_w, o.internal_h = int(internal[0]), int(internal[1])
    o.k = k
    o.repair = repair
    o.render = _RENDER[render]
    if static_protection:
        o.keep_static = 1
        if isinstance(static_protection, dict):
            o.static_frames = int(static_protection.get("frames", o.static_frames))
            o.static_tolerance = int(static_protection.get("tolerance", o.static_tolerance))
    o.fast_render = int(fast_render)
    o.linear_t = int(linear_t)
    o.keep_maps = int(keep_maps)
    o.protection_mask = int(protection_mask)
    o._tune = tune.encode() if tune else None
    o.tune = o._tune
    return o


class Session:
    """The plans of a model for one frame size: Session(engine, model, width, height, internal="auto"|"frame"|(w, h),
    k=1, repair=1, render="model"|"bilinear"|"bicubic", static_protection=None|True|{"frames", "tolerance"},
    fast_render=False, linear_t=False, keep_maps=False, protection_mask=False, tune=None)."""

    def __init__(self, engine: Engine, model: Model, width: int, height: int, **options):
        self._o = _session_options(width, height, **options)
        self.width, self.height = width, height
        self._e = engine
        self._h = _load().fh_session_create(engine._h, model._h, byref(self._o))
        if not self._h:
            raise _error()

    def add_input(self) -> int:
        return _check(_load().fh_session_add_input(self._h))

    def add_output(self) -> int:
        return _check(_load().fh_session_add_output(self._h))

    def write(self, slot: int, rgba) -> None:
        arr, n = _bytes(rgba)
        _check(_load().fh_session_write_rgba(self._h, slot, arr, n))

    def write_mask(self, mask) -> None:
        """The protection mask (protection_mask=True): width * height bytes, 0 interpolates; applies from the next pair."""
        arr, n = _bytes(mask)
        _check(_load().fh_session_write_mask(self._h, arr, n))

    def read(self, slot: int) -> bytes:
        n = self.width * self.height * 4
        buf = (c_uint8 * n)()
        _check(_load().fh_session_read_rgba(self._h, slot, buf, n))
        return bytes(buf)

    def pair(self, a: int, b: int) -> None:
        _check(_load().fh_session_pair(self._h, a, b))

    def mid(self, t: float, out: int) -> None:
        _check(_load().fh_session_mid(self._h, t, out))

    def generate(self, a: int, b: int, t: float, out: int) -> None:
        _check(_load().fh_session_generate(self._h, a, b, t, out))

    def pair_timed(self, a: int, b: int) -> Optional[float]:
        """GPU milliseconds of the pair stage, or None without timestamp queries."""
        ms = c_double()
        return ms.value if _check(_load().fh_session_pair_timed(self._h, a, b, byref(ms))) == 1 else None

    def mid_timed(self, t: float, out: int) -> Optional[float]:
        ms = c_double()
        return ms.value if _check(_load().fh_session_mid_timed(self._h, t, out, byref(ms))) == 1 else None

    def warm_up(self) -> None:
        _check(_load().fh_session_warm_up(self._h))

    def report(self) -> dict:
        return _json(_load().fh_session_report_json, self._h)

    def close(self):
        if getattr(self, "_h", None):
            _load().fh_session_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        self.close()


class Classification:
    """What Stream.classify found: kind ('motion' | 'cut' | 'hold'), mean, max (0..1), black, motion (the Pacer's units)."""

    def __init__(self, c: _Classification):
        self.kind = _KIND.get(c.kind, "motion")
        self.mean, self.max, self.black, self.motion = c.mean, c.max, bool(c.black), c.motion

    def __repr__(self):
        return f"Classification({self.kind}, mean={self.mean:.4f}, max={self.max:.4f}, motion={self.motion:.1f})"


class Push:
    """What Stream.push opened: ready, base_slot, stats_pending, start_pts, end_pts."""

    def __init__(self, p: _Push):
        self.ready, self.base_slot, self.stats_pending = bool(p.ready), p.base_slot, bool(p.stats_pending)
        self.start_pts, self.end_pts = p.start_pts, p.end_pts

    def __repr__(self):
        return f"Push(ready={self.ready}, base_slot={self.base_slot}, {self.start_pts}..{self.end_pts})"


class Stream:
    """Frames with timestamps in, intervals out: Stream(engine, model, width, height, mode="realtime"|"video",
    cadence=False, scene_cut=0.08 (0 = never), held_frames=False|True|{"mean", "max"}, inputs=4, outputs=4, plus the
    Session options)."""

    def __init__(self, engine: Engine, model: Model, width: int, height: int, mode: str = "realtime",
                 cadence: bool = False, scene_cut: float = 0.08, held_frames=False, inputs: int = 4, outputs: int = 4,
                 **session_options):
        lib = _load()
        self._so = _session_options(width, height, **session_options)
        o = lib.fh_stream_options_default()
        o.mode = _MODE[mode]
        o.cadence = int(cadence)
        o.scene_cut = float(scene_cut)
        o.held_frames = int(bool(held_frames))
        if isinstance(held_frames, dict):
            o.held_mean = float(held_frames.get("mean", 0.0))
            o.held_max = float(held_frames.get("max", 0.0))
        o.inputs, o.outputs = inputs, outputs
        self.width, self.height = width, height
        self._e = engine
        self._h = lib.fh_stream_create(engine._h, model._h, byref(self._so), byref(o))
        if not self._h:
            raise _error()

    def add_input(self) -> int:
        return _check(_load().fh_stream_add_input(self._h, self._e._h))

    def add_output(self) -> int:
        return _check(_load().fh_stream_add_output(self._h, self._e._h))

    def next_input(self, in_use: Sequence[int] = ()) -> Optional[int]:
        """The slot to write the next frame into, or None when the ring is exhausted."""
        arr = (c_uint32 * max(1, len(in_use)))(*in_use)
        slot = _check(_load().fh_stream_next_input(self._h, arr, len(in_use))) if True else -1
        return None if slot < 0 else slot

    def write(self, slot: int, rgba) -> None:
        arr, n = _bytes(rgba)
        _check(_load().fh_stream_write_rgba(self._h, self._e._h, slot, arr, n))

    def write_mask(self, mask) -> None:
        arr, n = _bytes(mask)
        _check(_load().fh_stream_write_mask(self._h, self._e._h, arr, n))

    def classify(self, previous: int, slot: int) -> Optional[Classification]:
        """The pair's kind before pushing it; None when every readback is in flight (take it as motion)."""
        c = _Classification()
        return Classification(c) if _check(_load().fh_stream_classify(self._h, self._e._h, previous, slot, byref(c))) == 1 else None

    def push(self, slot: int, pts_ms: float) -> Push:
        p = _Push()
        _check(_load().fh_stream_push(self._h, self._e._h, slot, pts_ms, byref(p)))
        return Push(p)

    def resolve_stats(self) -> None:
        _check(_load().fh_stream_resolve_stats(self._h, self._e._h))

    def real_replaced(self) -> bool:
        return bool(_check(_load().fh_stream_real_replaced(self._h)))

    def generate(self, t: float, out: int) -> bool:
        """True when the output slot holds the frame, False when the source frame is enough (t 0, no re-timing)."""
        return _check(_load().fh_stream_generate(self._h, self._e._h, t, out)) == 1

    def read(self, out: int) -> bytes:
        n = self.width * self.height * 4
        buf = (c_uint8 * n)()
        _check(_load().fh_stream_read_rgba(self._h, self._e._h, out, buf, n))
        return bytes(buf)

    def warm_up(self) -> None:
        _check(_load().fh_stream_warm_up(self._h, self._e._h))

    def reset(self) -> None:
        _check(_load().fh_stream_reset(self._h))

    def mouse(self, t: float, dx: float, dy: float) -> None:
        """Mouse counts at time t for the camera warp; t on the clock of push's pts (then arrival times in host ms)."""
        _check(_load().fh_stream_mouse(self._h, self._e._h, t, dx, dy))

    def set_mouse_captured(self, captured: bool) -> None:
        _check(_load().fh_stream_set_mouse_captured(self._h, int(bool(captured))))

    def warp(self, slot: int, out: int, now: float, output: bool = False) -> bool:
        """The frame about to be presented at now (input slot, or output slot with output=True) turned by the mouse
        motion since, into output slot out. True: present out; False: the warp is off, present the frame as it is."""
        return _check(_load().fh_stream_warp(self._h, self._e._h, int(output), slot, out, now)) == 1

    def report(self) -> dict:
        return _json(_load().fh_stream_report_json, self._h)

    def close(self):
        if getattr(self, "_h", None):
            _load().fh_stream_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        self.close()


class Request:
    """A frame to generate: phase t of the pair (previous, current), due at due_ms; retime and probe flags."""

    def __init__(self, r: _Request):
        self.id, self.previous, self.current, self.t = r.id, r.previous, r.current, r.t
        self.due_ms, self.retime, self.probe = r.due_ms, bool(r.retime), bool(r.probe)

    def __repr__(self):
        return f"Request({self.id}, t={self.t:.3f}, due={self.due_ms:.1f}, pair=({self.previous}, {self.current}))"


class Present:
    """What to show now: frame ('source' | 'output'), slot, due_ms, late_ms, mid, fallback."""

    def __init__(self, t: _Tick):
        self.frame = "source" if t.frame_kind == 0 else "output"
        self.slot, self.due_ms, self.late_ms = t.frame_id, t.due_ms, t.late_ms
        self.mid, self.fallback = bool(t.mid), bool(t.fallback)


class Tick:
    """present (a Present or None), generate (a list of Request), factor, dropped."""

    def __init__(self, t: _Tick, requests):
        self.present = Present(t) if t.has_present else None
        self.generate = requests
        self.factor, self.dropped = t.factor, t.dropped


class Pacer:
    """The scheduler between source frames and display ticks (ARCHITECTURE.md). Pacer(mode="auto" | int |
    "display_hz" | ("target", fps), **options) with the option names of PacerOptions; times are the host's
    monotonic milliseconds."""

    def __init__(self, mode="auto", **options):
        lib = _load()
        o = lib.fh_pacer_options_default()
        self._apply_mode(o, mode)
        for k, v in options.items():
            if not hasattr(o, k):
                raise TypeError(f"unknown Pacer option '{k}'")
            setattr(o, k, int(v) if isinstance(getattr(o, k), int) else float(v))
        self._h = lib.fh_pacer_create(byref(o))
        if not self._h:
            raise _error()
        self._reqs = (_Request * 256)()

    @staticmethod
    def _apply_mode(o: PacerOptions, mode):
        if mode == "auto":
            o.mode = 0
        elif mode == "display_hz":
            o.mode = 2
        elif isinstance(mode, int):
            o.mode, o.factor = 1, mode
        elif isinstance(mode, tuple) and mode[0] == "target":
            o.mode, o.target_fps = 3, float(mode[1])
        else:
            raise ValueError("mode is 'auto', an integer factor, 'display_hz' or ('target', fps)")

    def _requests(self, n: int):
        got = _check(_load().fh_pacer_requests(self._h, self._reqs, len(self._reqs)))
        return [Request(self._reqs[i]) for i in range(min(got, len(self._reqs)))]

    def tick(self, now_ms: float) -> Tick:
        t = _Tick()
        _check(_load().fh_pacer_tick(self._h, now_ms, byref(t)))
        return Tick(t, self._requests(t.requests) if t.requests else [])

    def source_frame(self, now_ms: float, slot: int, pts_ms: float, kind: str = "motion", motion: Optional[float] = None,
                     skipped: int = 0) -> list:
        """A source frame with its kind; returns the requests to generate before the pair changes."""
        f = _SourceFrame(slot, pts_ms, _KIND_CODE[kind], int(motion is not None), float(motion or 0.0), skipped)
        n = _check(_load().fh_pacer_source_frame(self._h, now_ms, byref(f)))
        return self._requests(n) if n else []

    def generated(self, now_ms: float, request: int, output: int) -> None:
        _check(_load().fh_pacer_generated(self._h, now_ms, request, output))

    def skipped(self, request: int) -> None:
        _check(_load().fh_pacer_skipped(self._h, request))

    def cost_sample(self, now_ms: float, stage: str, ms: float) -> bool:
        """Feeds a measured GPU time ('pair' | 'mid' | 'present'); False when ignored inside the settle window."""
        return _check(_load().fh_pacer_cost_sample(self._h, now_ms, _STAGE[stage], ms)) == 1

    def host_stall(self, now_ms: float, ms: float) -> None:
        _check(_load().fh_pacer_host_stall(self._h, now_ms, ms))

    def mode_change(self, now_ms: float) -> None:
        _check(_load().fh_pacer_mode_change(self._h, now_ms))

    def reset(self, now_ms: float) -> None:
        _check(_load().fh_pacer_reset(self._h, now_ms))

    def set_options(self, now_ms: float, mode="auto", **options) -> None:
        o = _load().fh_pacer_options_default()
        self._apply_mode(o, mode)
        for k, v in options.items():
            setattr(o, k, int(v) if isinstance(getattr(o, k), int) else float(v))
        _check(_load().fh_pacer_set_options(self._h, now_ms, byref(o)))

    def report(self) -> dict:
        return _json(_load().fh_pacer_report_json, self._h)

    def record(self, host: str = "") -> None:
        """Records every call from now on as a trace v1 (what flyhigh-dev replay reproduces); trace() reads it."""
        _check(_load().fh_pacer_record(self._h, host.encode()))

    def trace(self) -> dict:
        return _json(_load().fh_pacer_trace_json, self._h)

    def close(self):
        if getattr(self, "_h", None):
            _load().fh_pacer_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        self.close()
