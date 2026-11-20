//! Tensor buffers: creation, the per-plan arena that reuses dead tensors' buffers, scratch release, memory accounting.

use super::memory::{Buffer, Charge, Kind};
use crate::Gpu;
use std::collections::HashSet;
use std::rc::Rc;
use wgpu::util::DeviceExt;

// FH_ARENA_TRACE=1: every arena decision on stderr (plan building only; a static plan never allocates afterwards)
pub(crate) fn arena_trace() -> bool {
    std::env::var_os("FH_ARENA_TRACE").is_some()
}

#[derive(Default)]
pub(crate) struct Workspace {
    slots: Vec<(u64, Option<Buffer>)>,
    allocated: Vec<Buffer>,
    mismatch: bool,
}

impl Workspace {
    fn reuse(&mut self, size: u64) -> Option<Buffer> {
        if self.slots.is_empty() || self.mismatch {
            return None;
        }
        match self.slots.get(self.allocated.len()) {
            Some((len, buf)) if *len == size => buf.clone(),
            _ => {
                self.mismatch = true;
                None
            }
        }
    }

    pub(crate) fn finish(&mut self, retained: &HashSet<*const Charge>, used: &HashSet<*const Charge>) -> bool {
        if self.slots.is_empty() {
            self.slots = self
                .allocated
                .drain(..)
                .map(|b| {
                    let id = Rc::as_ptr(&b.charge);
                    (b.size(), (used.contains(&id) && !retained.contains(&id)).then_some(b))
                })
                .collect();
            return true;
        }
        // Compare the actual liveness of this build, not just its allocation sizes.
        let ok = !self.mismatch
            && self.slots.len() == self.allocated.len()
            && self.slots.iter().zip(&self.allocated).all(|((_, shared), actual)| {
                shared.as_ref().is_none_or(|_| !retained.contains(&Rc::as_ptr(&actual.charge)))
            });
        self.allocated.clear();
        ok
    }
}

impl Gpu {
    pub fn buf(&self, data: &[f32]) -> Rc<Buffer> {
        // f16 storage
        let h: Vec<u16> =
            if self.f32_mode { Vec::new() } else { data.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect() };
        let bytes: &[u8] = if self.f32_mode { bytemuck::cast_slice(data) } else { bytemuck::cast_slice(&h) };
        let raw = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fh_const"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Rc::new(self.memory.buffer(Kind::InitializedBuffers, raw))
    }
    /// Bytes of tensor buffers still alive (the engine's own video memory, without textures).
    pub fn live_bytes(&self) -> u64 {
        let report = self.memory.report();
        report.tensors.live_bytes + report.cadence_scratch.live_bytes
    }
    /// Sizes of the live tensor buffers, largest first (diagnostics).
    pub fn live_sizes(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.live.borrow().iter().filter_map(|w| w.upgrade()).map(|b| b.bytes).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v
    }
}

impl Gpu {
    pub fn alloc(&self, n: usize) -> Rc<Buffer> {
        self.alloc_raw((n * self.esize()) as u64)
    } // n tensor elements
    /// bytes per tensor element: 2 (f16) or 4 (the f32 path)
    pub fn esize(&self) -> usize {
        if self.f32_mode {
            4
        } else {
            2
        }
    }
    pub fn alloc_f32(&self, n: usize) -> Rc<Buffer> {
        self.alloc_raw((n * 4) as u64)
    } // n f32 elements (split-K partials)
      // kernel scratch (shifted copies, split-K partials, materialized views) lives only inside one op: note it here and
      // `release_scratch` hands it to the arena once the op is built
    pub fn note_scratch(&self, b: &Rc<Buffer>) {
        if self.arena.borrow().is_some() {
            self.scratch.borrow_mut().push(b.clone());
        }
    }
    pub fn release_scratch(&self) {
        let v = std::mem::take(&mut *self.scratch.borrow_mut());
        for b in v {
            self.release(b);
        }
    }
    pub fn alloc_raw(&self, size: u64) -> Rc<Buffer> {
        if let Some(free) = self.arena.borrow_mut().as_mut() {
            // smallest released buffer that fits (exact size first); kernels address `n` elements, not the buffer size
            if let Some(i) = free.iter().position(|b| b.size() == size).or_else(|| {
                free.iter().enumerate().filter(|(_, b)| b.size() >= size).min_by_key(|(_, b)| b.size()).map(|(i, _)| i)
            }) {
                if arena_trace() {
                    eprintln!(
                        "arena: alloc {size} <- reuse {} (free: {:?})",
                        free[i].size(),
                        free.iter().map(|b| b.size()).collect::<Vec<_>>()
                    );
                }
                return free.swap_remove(i);
            }
            if arena_trace() {
                eprintln!("arena: alloc {size} <- new (free: {:?})", free.iter().map(|b| b.size()).collect::<Vec<_>>());
            }
        }
        let mut workspace = self.workspace.borrow_mut();
        let b = workspace
            .as_mut()
            .and_then(|w| w.reuse(size))
            .map(Rc::new)
            .unwrap_or_else(|| self.create_tensor(size, Kind::Tensors));
        if let Some(w) = workspace.as_mut() {
            // Clone the raw handle, not Rc<Buffer>, so intra-plan last-use checks are unchanged.
            w.allocated.push((*b).clone());
        }
        if self.arena.borrow().is_some() {
            self.arena_ids.borrow_mut().insert(Rc::as_ptr(&b) as usize, Rc::downgrade(&b));
        }
        b
    }

    pub(crate) fn alloc_cadence(&self, size: u64) -> Rc<Buffer> {
        self.create_tensor(size, Kind::CadenceScratch)
    }

    fn create_tensor(&self, size: u64, kind: Kind) -> Rc<Buffer> {
        self.alloc_bytes.set(self.alloc_bytes.get() + size);
        let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fh_tensor"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b = Rc::new(self.memory.buffer(kind, raw));
        let mut live = self.live.borrow_mut();
        if live.len() == live.capacity() {
            live.retain(|w| w.strong_count() > 0);
        }
        live.push(Rc::downgrade(&b.charge));
        b
    }
    // give a dead tensor's buffer back to the arena (only `alloc` buffers, and only when nothing else holds the Rc:
    // views, aliases, copy steps). Constant buffers (`buf`) never enter: they are written once and read every frame.
    pub fn release(&self, b: Rc<Buffer>) {
        let (count, known) = (
            Rc::strong_count(&b),
            self.arena_ids.borrow().get(&(Rc::as_ptr(&b) as usize)).is_some_and(|w| w.as_ptr() == Rc::as_ptr(&b)),
        );
        if arena_trace() {
            eprintln!(
                "arena: release {} -> {}",
                b.size(),
                if count != 1 {
                    format!("kept (count {count})")
                } else if !known {
                    "kept (not arena)".into()
                } else {
                    "free".into()
                }
            );
        }
        if count != 1 || !known {
            return;
        }
        if let Some(free) = self.arena.borrow_mut().as_mut() {
            free.push(b);
        }
    }
}
