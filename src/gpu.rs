/// GPU compute pipeline for fractal rendering via wgpu.

use crate::fractals::{FractalParams, GpuParams};

/// Align width so bytes_per_row (width * 4) is a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
fn align_width(w: u32) -> u32 {
    let align = 256 / 4; // 64 pixels
    (w + align - 1) / align * align
}

/// Holds all wgpu state: device, pipelines, buffers, textures.
pub struct GpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    // Compute pipelines
    escape_pipeline: wgpu::ComputePipeline,
    newton_pipeline: wgpu::ComputePipeline,
    colorize_pipeline: wgpu::ComputePipeline,

    // Buffers
    params_buffer: wgpu::Buffer,
    iterations_buffer: wgpu::Buffer,
    final_z_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    roots_buffer: wgpu::Buffer,

    // Bind groups (rebuilt on resize)
    escape_bind_group: wgpu::BindGroup,
    newton_bind_group: wgpu::BindGroup,
    colorize_bind_group: wgpu::BindGroup,

    // Bind group layouts (needed for rebuilding)
    escape_bind_group_layout: wgpu::BindGroupLayout,
    newton_bind_group_layout: wgpu::BindGroupLayout,
    colorize_bind_group_layout: wgpu::BindGroupLayout,

    // Display texture
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub texture_id: Option<egui::TextureId>,

    // Current dimensions
    pub width: u32,
    pub height: u32,

    // Timing
    pub last_render_ms: f64,
    pub gpu_name: String,
}

impl GpuState {
    pub fn new(render_state: &eframe::egui_wgpu::RenderState) -> Self {
        let device = &render_state.device;
        let queue = &render_state.queue;
        let adapter_info = render_state.adapter.get_info();
        let gpu_name = adapter_info.name.clone();

        let width = align_width(960);
        let height = 720u32;
        let num_pixels = (width * height) as u64;

        // Create shader modules
        let escape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("escape compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/escape.wgsl").into()),
        });
        let newton_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("newton compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/newton.wgsl").into()),
        });
        let colorize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("colorize"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/colorize.wgsl").into()),
        });

        // Uniform buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Storage buffers
        let iterations_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iterations"),
            size: num_pixels * 4, // f32 per pixel
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let final_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("final_z"),
            size: num_pixels * 8, // vec2<f32> per pixel
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: num_pixels * 4, // u32 (packed RGBA) per pixel
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let roots_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roots"),
            size: 8 * 16, // up to 16 roots, vec2<f32> each
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layouts
        let escape_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("escape layout"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry_storage(1, false), // iterations (read_write)
                    bgl_entry_storage(2, false), // final_z (read_write)
                ],
            });

        let newton_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("newton layout"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry_storage(1, false), // iterations
                    bgl_entry_storage(2, false), // final_z
                    bgl_entry_storage(3, true),  // roots (read_only)
                ],
            });

        let colorize_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("colorize layout"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry_storage(1, true),  // iterations (read_only)
                    bgl_entry_storage(2, true),  // final_z (read_only)
                    bgl_entry_storage(3, false), // output (read_write)
                    bgl_entry_storage(4, true),  // roots (read_only)
                ],
            });

        // Compute pipelines
        let escape_pipeline = create_pipeline(device, &escape_shader, &escape_bind_group_layout);
        let newton_pipeline = create_pipeline(device, &newton_shader, &newton_bind_group_layout);
        let colorize_pipeline =
            create_pipeline(device, &colorize_shader, &colorize_bind_group_layout);

        // Bind groups
        let escape_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("escape bg"),
            layout: &escape_bind_group_layout,
            entries: &[
                bg_entry(0, &params_buffer),
                bg_entry(1, &iterations_buffer),
                bg_entry(2, &final_z_buffer),
            ],
        });

        let newton_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("newton bg"),
            layout: &newton_bind_group_layout,
            entries: &[
                bg_entry(0, &params_buffer),
                bg_entry(1, &iterations_buffer),
                bg_entry(2, &final_z_buffer),
                bg_entry(3, &roots_buffer),
            ],
        });

        let colorize_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("colorize bg"),
            layout: &colorize_bind_group_layout,
            entries: &[
                bg_entry(0, &params_buffer),
                bg_entry(1, &iterations_buffer),
                bg_entry(2, &final_z_buffer),
                bg_entry(3, &output_buffer),
                bg_entry(4, &roots_buffer),
            ],
        });

        // Display texture
        let (texture, texture_view) = create_texture(device, width, height);

        GpuState {
            device: device.clone(),
            queue: queue.clone(),
            escape_pipeline,
            newton_pipeline,
            colorize_pipeline,
            params_buffer,
            iterations_buffer,
            final_z_buffer,
            output_buffer,
            roots_buffer,
            escape_bind_group,
            newton_bind_group,
            colorize_bind_group,
            escape_bind_group_layout,
            newton_bind_group_layout,
            colorize_bind_group_layout,
            texture,
            texture_view,
            texture_id: None,
            width,
            height,
            last_render_ms: 0.0,
            gpu_name,
        }
    }

    /// Resize GPU buffers and texture to new dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = align_width(width);
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        let num_pixels = (width * height) as u64;

        self.iterations_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iterations"),
            size: num_pixels * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.final_z_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("final_z"),
            size: num_pixels * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: num_pixels * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let (texture, texture_view) = create_texture(&self.device, width, height);
        self.texture = texture;
        self.texture_view = texture_view;
        self.texture_id = None; // Force re-registration with egui

        // Rebuild bind groups with new buffers
        self.escape_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("escape bg"),
            layout: &self.escape_bind_group_layout,
            entries: &[
                bg_entry(0, &self.params_buffer),
                bg_entry(1, &self.iterations_buffer),
                bg_entry(2, &self.final_z_buffer),
            ],
        });
        self.newton_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("newton bg"),
            layout: &self.newton_bind_group_layout,
            entries: &[
                bg_entry(0, &self.params_buffer),
                bg_entry(1, &self.iterations_buffer),
                bg_entry(2, &self.final_z_buffer),
                bg_entry(3, &self.roots_buffer),
            ],
        });
        self.colorize_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("colorize bg"),
            layout: &self.colorize_bind_group_layout,
            entries: &[
                bg_entry(0, &self.params_buffer),
                bg_entry(1, &self.iterations_buffer),
                bg_entry(2, &self.final_z_buffer),
                bg_entry(3, &self.output_buffer),
                bg_entry(4, &self.roots_buffer),
            ],
        });
    }

    /// Run the fractal compute + colorize pipeline, update the display texture.
    pub fn render(&mut self, params: &FractalParams) {
        let start = std::time::Instant::now();

        let gpu_params = params.to_gpu_params(self.width, self.height);
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&gpu_params));

        // Upload roots if needed
        if params.fractal_type.needs_roots() {
            let roots = params.compute_roots();
            let roots_flat: Vec<f32> = roots.iter().flat_map(|r| r.iter().copied()).collect();
            // Pad to 16 roots (buffer size)
            let mut padded = roots_flat;
            padded.resize(32, 0.0); // 16 roots * 2 floats
            self.queue
                .write_buffer(&self.roots_buffer, 0, bytemuck::cast_slice(&padded));
        }

        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fractal"),
            });

        // Pass 1: Fractal iteration
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("iterate"),
                timestamp_writes: None,
            });
            if params.fractal_type.is_escape_time() {
                pass.set_pipeline(&self.escape_pipeline);
                pass.set_bind_group(0, &self.escape_bind_group, &[]);
            } else {
                pass.set_pipeline(&self.newton_pipeline);
                pass.set_bind_group(0, &self.newton_bind_group, &[]);
            }
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 2: Colorize
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("colorize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.colorize_pipeline);
            pass.set_bind_group(0, &self.colorize_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy output buffer → texture
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.width),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        self.last_render_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    /// Register the texture with egui for display.
    pub fn ensure_texture_registered(
        &mut self,
        renderer: &mut eframe::egui_wgpu::Renderer,
    ) {
        if self.texture_id.is_none() {
            let id = renderer.register_native_texture(
                &self.device,
                &self.texture_view,
                wgpu::FilterMode::Linear,
            );
            self.texture_id = Some(id);
        }
    }

    /// Update the egui texture after a render.
    pub fn update_texture(&mut self, renderer: &mut eframe::egui_wgpu::Renderer) {
        if let Some(id) = self.texture_id {
            renderer.update_egui_texture_from_wgpu_texture(
                &self.device,
                &self.texture_view,
                wgpu::FilterMode::Linear,
                id,
            );
        }
    }

    /// Export the current fractal to a PNG file.
    pub fn export_png(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let num_pixels = (self.width * self.height) as u64;
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: num_pixels * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.width),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = slice.get_mapped_range();
        let img =
            image::RgbaImage::from_raw(self.width, self.height, data.to_vec())
                .ok_or("Failed to create image")?;
        img.save(path)?;
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_entry_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    bgl_entry(
        binding,
        wgpu::BufferBindingType::Storage { read_only },
    )
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fractal output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
