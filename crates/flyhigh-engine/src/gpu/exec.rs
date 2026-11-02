//! Recording a plan into a command encoder and submitting it (pass and submit splitting for experiments).

use crate::{Gpu, Step};

impl Gpu {
    pub fn encode_plan(&self, enc: &mut wgpu::CommandEncoder, plan: &[Step]) {
        self.encode_plan_ts(enc, plan, None)
    }
    // with `qs`: pass-level timestamps (the WebGPU-standard kind) at the beginning of the first compute pass
    // (query 0) and the end of the last one (query 1) — the whole plan's GPU time without encoder-level queries
    pub fn encode_plan_ts(&self, enc: &mut wgpu::CommandEncoder, plan: &[Step], qs: Option<&wgpu::QuerySet>) {
        let last_c = plan.iter().rposition(|s| matches!(s, Step::Compute { .. }));
        let split = self.pass_split.get();
        let mut first = true;
        let mut i = 0;
        while i < plan.len() {
            match &plan[i] {
                Step::Copy { src, so, dst, dof, size } => {
                    enc.copy_buffer_to_buffer(src, *so, dst, *dof, *size);
                    i += 1;
                }
                Step::Compute { .. } => {
                    let j = (i..plan.len()).find(|&k| matches!(plan[k], Step::Copy { .. })).unwrap_or(plan.len()); // computes run to the next copy
                    let end = if split > 0 { j.min(i + split) } else { j }; // this pass = plan[i..end)
                    let is_last = last_c.is_some_and(|l| l < end);
                    let tw = match qs {
                        Some(q) if first || is_last => Some(wgpu::ComputePassTimestampWrites {
                            query_set: q,
                            beginning_of_pass_write_index: if first { Some(0) } else { None },
                            end_of_pass_write_index: if is_last { Some(1) } else { None },
                        }),
                        _ => None,
                    };
                    first = false;
                    let mut cp =
                        enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: tw });
                    while i < end {
                        if let Step::Compute { pipe, bg, groups, .. } = &plan[i] {
                            cp.set_pipeline(&self.pipeline(pipe));
                            cp.set_bind_group(0, bg, &[]);
                            cp.dispatch_workgroups(groups.0, groups.1, groups.2);
                            i += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }
    // Submit the plan: one command buffer, or (`submit_split` > 0) one per that many compute steps — a knob for the
    // shared-GPU latency experiments (a long single submission is preempted differently from several short ones).
    pub fn replay(&self, plan: &[Step]) {
        let ss = self.submit_split.get();
        let mut start = 0;
        while start < plan.len() {
            let end = if ss == 0 {
                plan.len()
            } else {
                let mut n = 0;
                let mut e = start;
                while e < plan.len() {
                    if matches!(plan[e], Step::Compute { .. }) {
                        n += 1;
                    }
                    e += 1;
                    if n >= ss {
                        break;
                    }
                }
                e
            };
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            self.encode_plan(&mut enc, &plan[start..end]);
            self.queue.submit(Some(enc.finish()));
            start = end;
        }
    }
}
