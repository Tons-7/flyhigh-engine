//! GPU time of a plan through timestamp queries: blocking, asynchronous and per-step profiles.

use crate::{map_read, Gpu, MapWait, Step};
use std::collections::HashMap;

impl Gpu {
    // pure GPU execution time of the whole command buffer, via timestamp queries (best of N).
    // wall_clock(replay+wait) - this = CPU-side overhead (encode + submit + driver launch + poll).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_time_ms(&self, plan: &[Step]) -> f64 {
        self.gpu_time_ms_n(plan, 60)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_time_ms_n(&self, plan: &[Step], reps: usize) -> f64 {
        pollster::block_on(self.gpu_time_ms_async(plan, reps))
    }
    // min-of-reps GPU time of `plan` from timestamp queries (encoder-level where available, else pass-level — the
    // WebGPU-standard kind, which Chrome exposes); NaN when the device has no timestamps at all
    pub async fn gpu_time_ms_async(&self, plan: &[Step], reps: usize) -> f64 {
        if !self.timestamps {
            return f64::NAN;
        }
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period = self.queue.get_timestamp_period() as f64; // ns per tick
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            if self.ts_inside {
                enc.write_timestamp(&qs, 0);
                self.encode_plan(&mut enc, plan);
                enc.write_timestamp(&qs, 1);
            } else {
                self.encode_plan_ts(&mut enc, plan, Some(&qs));
            }
            enc.resolve_query_set(&qs, 0..2, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &stage, 0, 16);
            self.queue.submit(Some(enc.finish()));
            let fut = map_read(&stage);
            #[cfg(not(target_arch = "wasm32"))]
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            if fut.await.is_err() {
                return f64::NAN;
            }
            let ms = {
                let data = stage.slice(..).get_mapped_range().unwrap();
                let ts: &[u64] = bytemuck::cast_slice(&data);
                ts[1].wrapping_sub(ts[0]) as f64 * period / 1e6
            };
            stage.unmap();
            if ms > 0.0 && ms < best {
                best = ms;
            }
        }
        best
    }
    // Timed run in two halves for hosts that must keep GPU submission order: `begin_timed` encodes `plan` with
    // timestamps, SUBMITS IT NOW and requests the readback; `TimedRun::finish` awaits the result later. (An async
    // export that submits inside its body runs that body in a microtask — after whatever the host submitted next.)
    pub fn begin_timed(&self, plan: &[Step]) -> Option<TimedRun> {
        if !self.timestamps {
            return None;
        }
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        if self.ts_inside {
            enc.write_timestamp(&qs, 0);
            self.encode_plan(&mut enc, plan);
            enc.write_timestamp(&qs, 1);
        } else {
            self.encode_plan_ts(&mut enc, plan, Some(&qs));
        }
        enc.resolve_query_set(&qs, 0..2, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &stage, 0, 16);
        self.queue.submit(Some(enc.finish()));
        let fut = map_read(&stage);
        Some(TimedRun { stage, fut, period: self.queue.get_timestamp_period() as f64 })
    }
    // per-step GPU time: one timestamp per step, each compute in its own pass so encoder-level
    // write_timestamp can bracket it; aggregated by pipeline name, averaged over REPS. Profiling only —
    // pass splitting adds inter-pass gaps, so the summed total exceeds replay time; the SHARES are the signal.
    pub fn profile_plan(&self, plan: &[Step]) -> Vec<(String, f64)> {
        if !self.ts_inside {
            println!("  (no encoder-level timestamp queries on this device: per-op profile unavailable)");
            return Vec::new();
        }
        let n = plan.len() as u32;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: n + 1,
        });
        let bytes = ((n + 1) * 8) as u64;
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period = self.queue.get_timestamp_period() as f64;
        const REPS: usize = 20;
        let mut acc: Vec<f64> = vec![0.0; plan.len()];
        for _ in 0..REPS {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for (i, st) in plan.iter().enumerate() {
                enc.write_timestamp(&qs, i as u32);
                match st {
                    Step::Copy { src, so, dst, dof, size } => enc.copy_buffer_to_buffer(src, *so, dst, *dof, *size),
                    Step::Compute { pipe, bg, groups, .. } => {
                        let mut cp = enc
                            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                        cp.set_pipeline(&self.pipeline(pipe));
                        cp.set_bind_group(0, bg, &[]);
                        cp.dispatch_workgroups(groups.0, groups.1, groups.2);
                    }
                }
            }
            enc.write_timestamp(&qs, n);
            enc.resolve_query_set(&qs, 0..n + 1, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &stage, 0, bytes);
            self.queue.submit(Some(enc.finish()));
            let slice = stage.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().unwrap().unwrap();
            {
                let data = slice.get_mapped_range().unwrap();
                let ts: &[u64] = bytemuck::cast_slice(&data);
                for i in 0..plan.len() {
                    acc[i] += ts[i + 1].wrapping_sub(ts[i]) as f64 * period / 1e6;
                }
            }
            stage.unmap();
        }
        let mut top: Vec<(usize, f64)> = acc.iter().enumerate().map(|(i, a)| (i, a / REPS as f64)).collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  top steps (idx pipe groups ms): naive conv groups = (ceil(w/8), ceil(h/8), cout); fused = (np/nblk, pixels/16)");
        for (i, ms) in top.iter().take(14) {
            match &plan[*i] {
                Step::Compute { pipe, groups, .. } => println!(
                    "    #{:<3} {:<8} ({:>4},{:>4},{:>4})  {:>6.3} ms",
                    i, pipe, groups.0, groups.1, groups.2, ms
                ),
                Step::Copy { size, .. } => println!("    #{:<3} copy     {:>10} B           {:>6.3} ms", i, size, ms),
            }
        }
        let mut by: HashMap<String, f64> = HashMap::new();
        for (i, st) in plan.iter().enumerate() {
            let key = match st {
                Step::Copy { .. } => "copy".to_string(),
                Step::Compute { pipe, .. } => pipe.to_string(),
            };
            *by.entry(key).or_insert(0.0) += acc[i] / REPS as f64;
        }
        let mut v: Vec<(String, f64)> = by.into_iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v
    }
}

// a submitted timed run whose timestamp readback is still in flight (see `Gpu::begin_timed`)
pub struct TimedRun {
    stage: wgpu::Buffer,
    fut: MapWait,
    period: f64,
}
impl TimedRun {
    pub async fn finish(self) -> f64 {
        if self.fut.await.is_err() {
            return f64::NAN;
        }
        let ms = {
            let data = self.stage.slice(..).get_mapped_range().unwrap();
            let ts: &[u64] = bytemuck::cast_slice(&data);
            ts[1].wrapping_sub(ts[0]) as f64 * self.period / 1e6
        };
        self.stage.unmap();
        ms
    }

    /// Native: poll the device and wait for the result.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn block(self, gpu: &Gpu) -> f64 {
        gpu.wait();
        pollster::block_on(self.finish())
    }
}
