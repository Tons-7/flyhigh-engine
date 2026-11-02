//! `Step`: one dispatch or copy of a plan, its bind group and metadata; plan dumps for the golden tests.

use crate::{Gpu, Tensor};
use std::rc::Rc;
use wgpu::util::DeviceExt;

pub enum Step {
    Compute { pipe: &'static str, bg: wgpu::BindGroup, groups: (u32, u32, u32), meta: Rc<StepMeta> },
    Copy { src: Rc<wgpu::Buffer>, so: u64, dst: Rc<wgpu::Buffer>, dof: u64, size: u64 },
}
pub type OpOutput = (Tensor, Vec<Step>);

/// What a compute step binds and its parameter bytes: a bind group is opaque once built, and the golden-plan tests
/// (`plan_dump`) need to see the structure of a plan to catch an accidental change.
pub struct StepMeta {
    pub bindings: Vec<Binding>,
    pub params: Vec<u8>,
}
pub enum Binding {
    Buffer(u64),
    Texture(u32, u32),
    Sampler,
    Params,
}
pub(crate) fn meta(bindings: Vec<Binding>, params: &[u8]) -> Rc<StepMeta> {
    Rc::new(StepMeta { bindings, params: params.to_vec() })
}
/// One line per step: pipeline, workgroup counts, every binding (buffer size, texture size, sampler, the parameter
/// uniform) and the parameter bytes as u32 words. The same plan always dumps the same text, so a dump is a golden.
pub fn plan_dump(plan: &[Step]) -> String {
    let mut out = String::new();
    for (i, s) in plan.iter().enumerate() {
        match s {
            Step::Copy { so, dof, size, .. } => out += &format!("{i:>3} copy {size} B (+{so} -> +{dof})\n"),
            Step::Compute { pipe, groups, meta, .. } => {
                let b: Vec<String> = meta
                    .bindings
                    .iter()
                    .map(|b| match b {
                        Binding::Buffer(n) => format!("buf{n}"),
                        Binding::Texture(w, h) => format!("tex{w}x{h}"),
                        Binding::Sampler => "sampler".into(),
                        Binding::Params => "params".into(),
                    })
                    .collect();
                let words: Vec<String> = meta
                    .params
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        format!("{:08x}", u32::from_le_bytes(w))
                    })
                    .collect();
                out += &format!(
                    "{i:>3} {pipe} ({},{},{}) [{}] {}\n",
                    groups.0,
                    groups.1,
                    groups.2,
                    b.join(" "),
                    words.join(" ")
                );
            }
        }
    }
    out
}

pub fn workgroups_2d(ow: u32, oh: u32, c: u32) -> (u32, u32, u32) {
    (ow.div_ceil(8), oh.div_ceil(8), c)
}

impl Gpu {
    // build a bind group (ins at 0..n, out at n, uniform at n+1, extras at n+2..); the bind group keeps its buffers alive
    pub fn bind(
        &self,
        pipe: &'static str,
        ins: &[&wgpu::Buffer],
        params: &[u8],
        out: &wgpu::Buffer,
        extra: &[&wgpu::Buffer],
    ) -> (wgpu::BindGroup, Rc<StepMeta>) {
        let pb = Rc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fh_params"),
            contents: params,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));
        *self.last_pb.borrow_mut() = Some(pb.clone());
        let mut e: Vec<wgpu::BindGroupEntry> = ins
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let n = ins.len() as u32;
        e.push(wgpu::BindGroupEntry { binding: n, resource: out.as_entire_binding() });
        e.push(wgpu::BindGroupEntry { binding: n + 1, resource: pb.as_entire_binding() });
        for (i, b) in extra.iter().enumerate() {
            e.push(wgpu::BindGroupEntry { binding: n + 2 + i as u32, resource: b.as_entire_binding() });
        }
        let mut bindings: Vec<Binding> = ins.iter().map(|b| Binding::Buffer(b.size())).collect();
        bindings.push(Binding::Buffer(out.size()));
        bindings.push(Binding::Params);
        bindings.extend(extra.iter().map(|b| Binding::Buffer(b.size())));
        (
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipeline(pipe).get_bind_group_layout(0),
                entries: &e,
            }),
            meta(bindings, params),
        )
    }
    pub fn compute_step(
        &self,
        pipe: &'static str,
        ins: &[&wgpu::Buffer],
        params: &[u8],
        out: &wgpu::Buffer,
        groups: (u32, u32, u32),
    ) -> Step {
        let (bg, meta) = self.bind(pipe, ins, params, out, &[]);
        Step::Compute { pipe, bg, groups, meta }
    }
    pub fn compute_step_ex(
        &self,
        pipe: &'static str,
        ins: &[&wgpu::Buffer],
        params: &[u8],
        out: &wgpu::Buffer,
        extra: &[&wgpu::Buffer],
        groups: (u32, u32, u32),
    ) -> Step {
        let (bg, meta) = self.bind(pipe, ins, params, out, extra);
        Step::Compute { pipe, bg, groups, meta }
    }
}

pub fn clone_step(s: &Step) -> Step {
    match s {
        Step::Compute { pipe, bg, groups, meta } => {
            Step::Compute { pipe, bg: bg.clone(), groups: *groups, meta: meta.clone() }
        }
        Step::Copy { src, so, dst, dof, size } => {
            Step::Copy { src: src.clone(), so: *so, dst: dst.clone(), dof: *dof, size: *size }
        }
    }
}
