//! Logical allocation sizes, independent of wgpu's allocator and GPU submission lifetimes.
use serde::Serialize;
use std::{cell::Cell, ops::Deref, rc::Rc};
use wgpu::util::DeviceExt;

/// Live and lifetime peak requested bytes for one resource category.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MemoryUsage {
    pub live_bytes: u64,
    pub peak_bytes: u64,
}

/// Logical GPU allocation sizes for this engine, excluding allocations owned by the host or driver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MemoryReport {
    pub live_bytes: u64,
    pub peak_bytes: u64,
    pub tensors: MemoryUsage,
    pub initialized_buffers: MemoryUsage,
    pub cadence_scratch: MemoryUsage,
    pub frame_textures: MemoryUsage,
    pub auxiliary_buffers: MemoryUsage,
    pub auxiliary_textures: MemoryUsage,
}

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Tensors,
    InitializedBuffers,
    CadenceScratch,
    FrameTextures,
    AuxiliaryBuffers,
    AuxiliaryTextures,
}

#[derive(Clone, Default)]
pub(crate) struct Memory(Rc<State>);

#[derive(Default)]
struct State {
    total: Cell<MemoryUsage>,
    categories: [Cell<MemoryUsage>; Kind::AuxiliaryTextures as usize + 1],
}

impl Memory {
    pub(crate) fn report(&self) -> MemoryReport {
        let total = self.0.total.get();
        let usage = |kind: Kind| self.0.categories[kind as usize].get();
        MemoryReport {
            live_bytes: total.live_bytes,
            peak_bytes: total.peak_bytes,
            tensors: usage(Kind::Tensors),
            initialized_buffers: usage(Kind::InitializedBuffers),
            cadence_scratch: usage(Kind::CadenceScratch),
            frame_textures: usage(Kind::FrameTextures),
            auxiliary_buffers: usage(Kind::AuxiliaryBuffers),
            auxiliary_textures: usage(Kind::AuxiliaryTextures),
        }
    }

    pub(crate) fn charge(&self, kind: Kind, bytes: u64) -> Charge {
        for counter in [&self.0.total, &self.0.categories[kind as usize]] {
            let mut usage = counter.get();
            usage.live_bytes += bytes;
            usage.peak_bytes = usage.peak_bytes.max(usage.live_bytes);
            counter.set(usage);
        }
        Charge { memory: self.clone(), kind, bytes }
    }

    pub(crate) fn buffer(&self, kind: Kind, raw: wgpu::Buffer) -> Buffer {
        let charge = self.charge(kind, raw.size());
        Buffer { raw, charge: Rc::new(charge) }
    }

    pub(crate) fn rgba8_slot(&self, texture: &wgpu::Texture) -> Charge {
        self.charge(Kind::FrameTextures, u64::from(texture.width()) * u64::from(texture.height()) * 4)
    }
}

pub(crate) struct Charge {
    memory: Memory,
    kind: Kind,
    pub(crate) bytes: u64,
}

#[cfg(test)]
impl Charge {
    pub(crate) fn is_tensor(&self) -> bool {
        matches!(self.kind, Kind::Tensors)
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        let state = &self.memory.0;
        for counter in [&state.total, &state.categories[self.kind as usize]] {
            let mut usage = counter.get();
            usage.live_bytes -= self.bytes;
            counter.set(usage);
        }
    }
}

/// An engine buffer. Dereferences to the underlying wgpu buffer for copies and bindings.
#[derive(Clone)]
pub struct Buffer {
    raw: wgpu::Buffer,
    // Plan steps share this charge without changing the arena's Rc<Buffer> liveness checks.
    pub(crate) charge: Rc<Charge>,
}

impl Deref for Buffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &wgpu::Buffer {
        &self.raw
    }
}

impl super::Gpu {
    pub(crate) fn buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Buffer {
        self.memory.buffer(Kind::AuxiliaryBuffers, self.device.create_buffer(desc))
    }

    pub(crate) fn buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor<'_>) -> Buffer {
        self.memory.buffer(Kind::AuxiliaryBuffers, self.device.create_buffer_init(desc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_is_the_maximum_overlapping_usage_even_without_sampling() {
        let memory = Memory::default();
        let first = Rc::new(memory.charge(Kind::Tensors, 10));
        let alias = first.clone();
        let scratch = memory.charge(Kind::CadenceScratch, 4);
        drop(first);
        drop(alias);
        assert_eq!(memory.report().live_bytes, 4);
        let slots = memory.charge(Kind::FrameTextures, 7);
        drop(scratch);
        drop(slots);
        let stopped = memory.report();
        assert_eq!(stopped.live_bytes, 0);
        assert_eq!(stopped.peak_bytes, 14);
        assert_eq!(stopped.frame_textures.peak_bytes, 7);
        assert_eq!(stopped.cadence_scratch.peak_bytes, 4);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires a GPU"]
    fn plan_bindings_keep_one_charge_without_blocking_arena_reuse() {
        use crate::{Engine, EngineOptions, Tensor};

        let engine = Engine::new(&EngineOptions::default()).unwrap();
        let gpu = engine.gpu();
        let baseline = engine.report().memory;

        *gpu.arena.borrow_mut() = Some(Vec::new());
        let src = Tensor { buf: gpu.alloc(64), c: 1, h: 8, w: 8, c0: 0, scale: 1.0 };
        let (dst, plan) = gpu.unary(&src, 0, 1.0, 0.0);
        let bytes = src.buf.size();
        let address = Rc::as_ptr(&src.buf);
        gpu.release(src.buf);
        let reused = gpu.alloc(64);
        assert_eq!(Rc::as_ptr(&reused), address);
        drop(reused);
        drop(dst);
        gpu.arena.borrow_mut().take();
        gpu.arena_ids.borrow_mut().clear();
        assert_eq!(engine.report().memory.tensors.live_bytes, bytes * 2);
        assert_eq!(gpu.live_sizes(), vec![bytes, bytes]);

        drop(plan);
        let stopped = engine.report().memory;
        assert_eq!(stopped.live_bytes, baseline.live_bytes);
        assert_eq!(stopped.tensors.peak_bytes, bytes * 2);
        assert!(gpu.live_sizes().is_empty());
    }
}
