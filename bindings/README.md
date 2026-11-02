# Bindings

The engine's surfaces by tier. Tier 1 is the engine itself: Rust (`flyhigh-engine`), C (`flyhigh-capi`,
`include/flyhigh.h`) and JavaScript (`flyhigh-web`). Tier 2 lives here, thin and written against nothing but the C
header: whatever the C ABI can do, these can.

| Binding | File | Shape |
|---|---|---|
| Python | `python/flyhigh.py` | ctypes, standard library only; classes `Engine`, `Model`, `Session`, `Stream`, `Pacer` with context managers; frames as any buffer (bytes, bytearray, numpy); errors as `FlyHighError` with the engine's message |
| C++ | `cpp/flyhigh.hpp` | header only, C++17; RAII handles `fh::Engine`, `fh::Model`, `fh::Session`, `fh::Stream`, `fh::Pacer`; errors as `fh::Error`; the C option structs with `options()` factories |

Both find the library the way their platform does: Python through `FLYHIGH_LIB` (a path), a file next to the module,
or the default name on the loader's path; C++ through the linker (`-lflyhigh` against `target/release`, see
`examples/c/CMakeLists.txt`). Build the library with `cargo build --release -p flyhigh-capi`.

```python
import flyhigh
with flyhigh.Engine() as engine, flyhigh.Model("models") as model:
    s = flyhigh.Session(engine, model, 1920, 1080, fast_render=True, linear_t=True)
    a, b, out = s.add_input(), s.add_input(), s.add_output()
    s.write(a, frame0); s.write(b, frame1)
    s.pair(a, b); s.mid(0.5, out)
    rgba = s.read(out)
```

```cpp
fh::Engine engine;
fh::Model model("models");
fh::Session s(engine, model, fh::Session::options(1920, 1080));
auto a = s.addInput(), b = s.addInput(), out = s.addOutput();
s.write(a, frame0); s.write(b, frame1);
s.pair(a, b); s.mid(0.5f, out);
std::vector<uint8_t> rgba = s.read(out);
```

Checks: `python bindings/python/test_flyhigh.py` and `bindings/cpp/example.cpp` run an in-between through the Session
and the Stream and a Pacer on synthetic ticks against the built library; CI runs them on lavapipe. Tier 3 (C#, Go and
others) is the community's, with the C header as the contract; a guide on writing one follows.
