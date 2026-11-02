//! Texture inputs and outputs: rgba8 texture views to tensors and back, texture creation.

use crate::ops::step::meta;
use crate::{workgroups_2d, Binding, Gpu, RgbaParams, Step, Tensor};
use wgpu::util::DeviceExt;

impl Gpu {
    // texture-input step: rgba8unorm texture view -> padded f16 CHW tensor
    pub fn tex_in_step(&self, view: &wgpu::TextureView, dst: &Tensor, w: u32, h: u32) -> Step {
        let rp = RgbaParams { w, h, wp: dst.w, hp: dst.h };
        let pb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fh_tex_in_params"),
            contents: bytemuck::bytes_of(&rp),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline("tex_in").get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: dst.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pb.as_entire_binding() },
            ],
        });
        Step::Compute {
            pipe: "tex_in",
            bg,
            groups: workgroups_2d(dst.w, dst.h, 1),
            meta: meta(
                vec![Binding::Texture(w, h), Binding::Buffer(dst.buf.size()), Binding::Params],
                bytemuck::bytes_of(&rp),
            ),
        }
    }
    // texture-output step: rendered tensor -> storage texture view (rgba8unorm, or bgra8unorm when `bgra`)
    pub fn tex_out_step(&self, src: &Tensor, view: &wgpu::TextureView, w: u32, h: u32, bgra: bool) -> Step {
        let pipe: &'static str = if bgra { "tex_out_bgra" } else { "tex_out" };
        let rp = RgbaParams { w, h, wp: src.w, hp: src.h };
        let pb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fh_tex_out_params"),
            contents: bytemuck::bytes_of(&rp),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline(pipe).get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 2, resource: pb.as_entire_binding() },
            ],
        });
        Step::Compute {
            pipe,
            bg,
            groups: workgroups_2d(w, h, 1),
            meta: meta(
                vec![Binding::Buffer(src.buf.size()), Binding::Texture(w, h), Binding::Params],
                bytemuck::bytes_of(&rp),
            ),
        }
    }
    // an rgba8unorm texture usable as a copyExternalImageToTexture destination, as a sampled input and as a copy source
    pub fn input_texture(&self, w: u32, h: u32) -> wgpu::Texture {
        self.input_texture_l(w, h, "fh_in")
    }
    pub fn input_texture_l(&self, w: u32, h: u32, label: &str) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }
    // an rgba8unorm storage texture (offscreen output target: timing, or a generated frame the host samples / copies later)
    pub fn output_texture(&self, w: u32, h: u32) -> wgpu::Texture {
        self.output_texture_l(w, h, "fh_out")
    }
    pub fn output_texture_l(&self, w: u32, h: u32, label: &str) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }
}
