/// GPU compute pipeline for fractal rendering via wgpu.
/// Multi-pass accumulation supersampling: each sub-pixel sample is rendered
/// at the output resolution with a coordinate offset, colorized, and
/// accumulated into a float buffer. A final pass normalizes and packs to RGBA.

use crate::fractals::{FractalParams, GpuParams, PerturbGpuParams};
use rug::Float;

/// Max sub-pixel samples for median mode. 9 (3×3) keeps the iterations buffer
/// under 128MB binding limit at 1920×1080 (9 * 1920 * 1080 * 4 = 75MB).
pub const MAX_MEDIAN_SAMPLES: u32 = 9;

/// Align width so bytes_per_row (width * 4) is a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
pub fn align_width(w: u32) -> u32 {
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
    finalize_pipeline: wgpu::ComputePipeline,
    median_finalize_pipeline: wgpu::ComputePipeline,

    // Buffers
    params_buffer: wgpu::Buffer,
    iterations_buffer: wgpu::Buffer,
    final_z_buffer: wgpu::Buffer,
    orbit_trap_buffer: wgpu::Buffer, // vec4<f32> per pixel — min distances to 4 trap points
    accum_buffer: wgpu::Buffer,     // vec4<f32> per pixel — accumulated color + weight
    output_buffer: wgpu::Buffer,
    roots_buffer: wgpu::Buffer,

    // Bind groups (rebuilt on resize)
    escape_bind_group: wgpu::BindGroup,
    newton_bind_group: wgpu::BindGroup,
    colorize_bind_group: wgpu::BindGroup,
    finalize_bind_group: wgpu::BindGroup,
    median_finalize_bind_group: wgpu::BindGroup,

    // Bind group layouts (needed for rebuilding)
    escape_bind_group_layout: wgpu::BindGroupLayout,
    newton_bind_group_layout: wgpu::BindGroupLayout,
    colorize_bind_group_layout: wgpu::BindGroupLayout,
    finalize_bind_group_layout: wgpu::BindGroupLayout,
    median_finalize_bind_group_layout: wgpu::BindGroupLayout,

    // Perturbation pipeline (for deep zoom Mandelbrot)
    perturb_pipeline: wgpu::ComputePipeline,
    perturb_bind_group_layout: wgpu::BindGroupLayout,
    perturb_bind_group: wgpu::BindGroup,
    ref_orbit_buffer: wgpu::Buffer,
    perturb_params_buffer: wgpu::Buffer,

    // CPU readback buffer (for egui::ColorImage upload)
    readback_buffer: wgpu::Buffer,

    // Current dimensions
    pub width: u32,           // aligned buffer stride (multiple of 64)
    pub display_width: u32,   // actual visible display width (texture width)
    pub height: u32,


    // Perturbation state
    pub using_perturbation: bool,
    ref_orbit_max_entries: u32, // current capacity of ref_orbit_buffer
    cached_ref_orbit: Option<(Float, Float, u32, u32, u32, [f32; 2])>, // (center_re, center_im, max_iter, orbit_len, fractal_type, julia_c)

    // Staging buffer for batched sample params (avoids per-sample GPU sync)
    params_staging_buffer: wgpu::Buffer,

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

        let display_width = 960u32;
        let width = align_width(display_width);
        let height = 720u32;
        let out_pixels = (width * height) as u64;

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
        let finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("finalize"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/finalize.wgsl").into()),
        });
        let median_finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("median_finalize"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/median_finalize.wgsl").into()),
        });

        // Uniform buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Storage buffers — all at output resolution
        let iterations_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iterations"),
            size: out_pixels * 4 * MAX_MEDIAN_SAMPLES as u64, // f32 per pixel × N samples for median
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let final_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("final_z"),
            size: out_pixels * 16, // vec4<f32> per pixel (z.xy, dz_mag, 0)
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let orbit_trap_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orbit_traps"),
            size: out_pixels * 16, // vec4<f32> per pixel (min dist to 4 trap points)
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accum"),
            size: out_pixels * 16, // vec4<f32> per pixel (rgb + weight)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: out_pixels * 4, // u32 (packed RGBA) per output pixel
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let roots_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roots"),
            size: 8 * 16, // up to 16 roots, vec2<f32> each
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Perturbation resources
        let perturb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("escape_perturb compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/escape_perturb.wgsl").into()),
        });
        let ref_orbit_max_entries = 50000u32;
        let ref_orbit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ref_orbit"),
            size: ref_orbit_max_entries as u64 * 16, // vec4<f32> per entry (double-single)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let perturb_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perturb_params"),
            size: std::mem::size_of::<PerturbGpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                    bgl_entry_storage(3, false), // orbit_traps (read_write)
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
                    bgl_entry_storage(4, false), // orbit_traps (read_write)
                ],
            });

        let colorize_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("colorize layout"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry_storage(1, true),  // iterations (read_only)
                    bgl_entry_storage(2, true),  // final_z (read_only)
                    bgl_entry_storage(3, false), // accum (read_write)
                    bgl_entry_storage(4, true),  // roots (read_only)
                    bgl_entry_storage(5, true),  // orbit_traps (read_only)
                ],
            });

        let finalize_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("finalize layout"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry_storage(1, true),  // accum (read_only)
                    bgl_entry_storage(2, false), // output (read_write)
                ],
            });

        let median_finalize_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("median_finalize layout"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry_storage(1, true),  // iterations (read_only)
                    bgl_entry_storage(2, false), // output (read_write)
                    bgl_entry_storage(3, true),  // final_z (read_only)
                    bgl_entry_storage(4, true),  // roots (read_only)
                    bgl_entry_storage(5, true),  // orbit_traps (read_only)
                ],
            });

        // Compute pipelines
        let escape_pipeline = create_pipeline(device, &escape_shader, &escape_bind_group_layout);
        let newton_pipeline = create_pipeline(device, &newton_shader, &newton_bind_group_layout);
        let colorize_pipeline =
            create_pipeline(device, &colorize_shader, &colorize_bind_group_layout);
        let finalize_pipeline =
            create_pipeline(device, &finalize_shader, &finalize_bind_group_layout);
        let median_finalize_pipeline =
            create_pipeline(device, &median_finalize_shader, &median_finalize_bind_group_layout);

        // Perturbation bind group layout + pipeline
        let perturb_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("perturb layout"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),   // params
                    bgl_entry_storage(1, false),                       // iterations (rw)
                    bgl_entry_storage(2, false),                       // final_z (rw)
                    bgl_entry_storage(3, true),                        // ref_orbit (read)
                    bgl_entry(4, wgpu::BufferBindingType::Uniform),   // perturb_params
                    bgl_entry_storage(5, false),                       // orbit_traps (rw)
                ],
            });
        let perturb_pipeline = create_pipeline(device, &perturb_shader, &perturb_bind_group_layout);

        // Bind groups
        let perturb_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("perturb bg"),
            layout: &perturb_bind_group_layout,
            entries: &[
                bg_entry(0, &params_buffer),
                bg_entry(1, &iterations_buffer),
                bg_entry(2, &final_z_buffer),
                bg_entry(3, &ref_orbit_buffer),
                bg_entry(4, &perturb_params_buffer),
                bg_entry(5, &orbit_trap_buffer),
            ],
        });

        let escape_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("escape bg"),
            layout: &escape_bind_group_layout,
            entries: &[
                bg_entry(0, &params_buffer),
                bg_entry(1, &iterations_buffer),
                bg_entry(2, &final_z_buffer),
                bg_entry(3, &orbit_trap_buffer),
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
                bg_entry(4, &orbit_trap_buffer),
            ],
        });

        let colorize_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("colorize bg"),
            layout: &colorize_bind_group_layout,
            entries: &[
                bg_entry(0, &params_buffer),
                bg_entry(1, &iterations_buffer),
                bg_entry(2, &final_z_buffer),
                bg_entry(3, &accum_buffer),
                bg_entry(4, &roots_buffer),
                bg_entry(5, &orbit_trap_buffer),
            ],
        });

        let finalize_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("finalize bg"),
            layout: &finalize_bind_group_layout,
            entries: &[
                bg_entry(0, &params_buffer),
                bg_entry(1, &accum_buffer),
                bg_entry(2, &output_buffer),
            ],
        });

        let median_finalize_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("median_finalize bg"),
            layout: &median_finalize_bind_group_layout,
            entries: &[
                bg_entry(0, &params_buffer),
                bg_entry(1, &iterations_buffer),
                bg_entry(2, &output_buffer),
                bg_entry(3, &final_z_buffer),
                bg_entry(4, &roots_buffer),
                bg_entry(5, &orbit_trap_buffer),
            ],
        });

        // CPU readback buffer (same size as output_buffer)
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_pixels * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Staging buffer for batched sample params (64 samples + 1 for finalize params)
        let params_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params_staging"),
            size: 65 * std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuState {
            device: device.clone(),
            queue: queue.clone(),
            escape_pipeline,
            newton_pipeline,
            colorize_pipeline,
            finalize_pipeline,
            params_buffer,
            iterations_buffer,
            final_z_buffer,
            orbit_trap_buffer,
            accum_buffer,
            output_buffer,
            roots_buffer,
            escape_bind_group,
            newton_bind_group,
            colorize_bind_group,
            finalize_bind_group,
            escape_bind_group_layout,
            newton_bind_group_layout,
            colorize_bind_group_layout,
            finalize_bind_group_layout,
            median_finalize_pipeline,
            median_finalize_bind_group,
            median_finalize_bind_group_layout,
            perturb_pipeline,
            perturb_bind_group_layout,
            perturb_bind_group,
            ref_orbit_buffer,
            perturb_params_buffer,
            readback_buffer,
            width,
            display_width,
            height,
            using_perturbation: false,
            ref_orbit_max_entries,
            cached_ref_orbit: None,
            params_staging_buffer,
            last_render_ms: 0.0,
            gpu_name,
        }
    }

    /// Resize GPU buffers to new dimensions.
    /// Buffers use aligned stride for wgpu row alignment.
    pub fn resize(&mut self, display_w: u32, height: u32) {
        let stride = align_width(display_w);
        if display_w == self.display_width && stride == self.width && height == self.height {
            return;
        }

        {
            self.display_width = display_w;
            self.width = stride;
            self.height = height;
            let out_pixels = (stride as u64) * (height as u64);

            self.iterations_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iterations"),
                size: out_pixels * 4 * MAX_MEDIAN_SAMPLES as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.final_z_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("final_z"),
                size: out_pixels * 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.orbit_trap_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orbit_traps"),
                size: out_pixels * 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.accum_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("accum"),
                size: out_pixels * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("output"),
                size: out_pixels * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: out_pixels * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Rebuild bind groups with new buffers
            self.escape_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("escape bg"),
                layout: &self.escape_bind_group_layout,
                entries: &[
                    bg_entry(0, &self.params_buffer),
                    bg_entry(1, &self.iterations_buffer),
                    bg_entry(2, &self.final_z_buffer),
                    bg_entry(3, &self.orbit_trap_buffer),
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
                    bg_entry(4, &self.orbit_trap_buffer),
                ],
            });
            self.colorize_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("colorize bg"),
                layout: &self.colorize_bind_group_layout,
                entries: &[
                    bg_entry(0, &self.params_buffer),
                    bg_entry(1, &self.iterations_buffer),
                    bg_entry(2, &self.final_z_buffer),
                    bg_entry(3, &self.accum_buffer),
                    bg_entry(4, &self.roots_buffer),
                    bg_entry(5, &self.orbit_trap_buffer),
                ],
            });
            self.finalize_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("finalize bg"),
                layout: &self.finalize_bind_group_layout,
                entries: &[
                    bg_entry(0, &self.params_buffer),
                    bg_entry(1, &self.accum_buffer),
                    bg_entry(2, &self.output_buffer),
                ],
            });
            self.median_finalize_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("median_finalize bg"),
                layout: &self.median_finalize_bind_group_layout,
                entries: &[
                    bg_entry(0, &self.params_buffer),
                    bg_entry(1, &self.iterations_buffer),
                    bg_entry(2, &self.output_buffer),
                    bg_entry(3, &self.final_z_buffer),
                    bg_entry(4, &self.roots_buffer),
                    bg_entry(5, &self.orbit_trap_buffer),
                ],
            });
            self.perturb_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("perturb bg"),
                layout: &self.perturb_bind_group_layout,
                entries: &[
                    bg_entry(0, &self.params_buffer),
                    bg_entry(1, &self.iterations_buffer),
                    bg_entry(2, &self.final_z_buffer),
                    bg_entry(3, &self.ref_orbit_buffer),
                    bg_entry(4, &self.perturb_params_buffer),
                    bg_entry(5, &self.orbit_trap_buffer),
                ],
            });
        }
    }

    /// Ensure ref_orbit_buffer is large enough for the given max_iter.
    fn ensure_ref_orbit_capacity(&mut self, max_iter: u32) {
        let needed = max_iter + 1; // +1 for the escaping value
        if needed > self.ref_orbit_max_entries {
            self.ref_orbit_max_entries = needed;
            self.ref_orbit_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ref_orbit"),
                size: needed as u64 * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // Rebuild perturbation bind group with new buffer
            self.perturb_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("perturb bg"),
                layout: &self.perturb_bind_group_layout,
                entries: &[
                    bg_entry(0, &self.params_buffer),
                    bg_entry(1, &self.iterations_buffer),
                    bg_entry(2, &self.final_z_buffer),
                    bg_entry(3, &self.ref_orbit_buffer),
                    bg_entry(4, &self.perturb_params_buffer),
                    bg_entry(5, &self.orbit_trap_buffer),
                ],
            });
        }
    }

    /// Run the fractal compute + colorize pipeline with multi-pass accumulation
    /// supersampling, then finalize and update the display texture.
    pub fn render(&mut self, params: &FractalParams, effective_ss: u32, use_median: bool) {
        let start = std::time::Instant::now();

        let ss = effective_ss;
        let wg_x = (self.display_width + 15) / 16;
        let wg_y = (self.height + 15) / 16;

        // Determine if we should use perturbation (all escape-time z²+c types at deep zoom)
        let pixel_step = params.pixel_step_x(self.display_width);
        let use_perturb = params.fractal_type.is_escape_time()
            && params.fractal_type != crate::fractals::FractalType::Multibrot
            && pixel_step < 1e-7;
        self.using_perturbation = use_perturb;

        // Decompose pixel_step into f32 mantissa + i32 exponent for extended-range perturbation
        let step_x = params.pixel_step_x(self.display_width);
        let step_y = params.pixel_step_y(self.height);
        let ps_exp = pixel_step.log2().floor() as i32;
        let scale = 2.0_f64.powi(ps_exp);
        let ps_mantissa_x = (step_x / scale) as f32;
        let ps_mantissa_y = (step_y / scale) as f32;

        // Upload reference orbit if using perturbation (cached to avoid recomputation)
        if use_perturb {
            let ft_idx = params.fractal_type.shader_index();
            let need_recompute = match &self.cached_ref_orbit {
                None => true,
                Some((prev_re, prev_im, prev_iter, _, prev_ft, prev_jc)) => {
                    *prev_re != params.center_re
                        || *prev_im != params.center_im
                        || *prev_iter < params.max_iter
                        || *prev_ft != ft_idx
                        || *prev_jc != params.julia_c
                }
            };

            if need_recompute {
                self.ensure_ref_orbit_capacity(params.max_iter);
                let julia_c = if params.fractal_type == crate::fractals::FractalType::Julia {
                    Some((params.julia_c[0] as f64, params.julia_c[1] as f64))
                } else {
                    None
                };
                let perturb_data = crate::fractals::compute_variant_reference_orbit(
                    &params.center_re, &params.center_im,
                    params.max_iter, pixel_step,
                    params.fractal_type, julia_c,
                );
                self.queue.write_buffer(
                    &self.ref_orbit_buffer,
                    0,
                    bytemuck::cast_slice(&perturb_data.orbit),
                );
                let perturb_gpu = PerturbGpuParams {
                    ref_orbit_len: perturb_data.orbit_len,
                    pixel_step_exp: ps_exp,
                    _pad: [0; 2],
                };
                self.queue.write_buffer(
                    &self.perturb_params_buffer,
                    0,
                    bytemuck::bytes_of(&perturb_gpu),
                );
                self.cached_ref_orbit = Some((params.center_re.clone(), params.center_im.clone(), params.max_iter, perturb_data.orbit_len, ft_idx, params.julia_c));
            } else {
                let orbit_len = self.cached_ref_orbit.as_ref().unwrap().3;
                let perturb_gpu = PerturbGpuParams {
                    ref_orbit_len: orbit_len,
                    pixel_step_exp: ps_exp,
                    _pad: [0; 2],
                };
                self.queue.write_buffer(
                    &self.perturb_params_buffer,
                    0,
                    bytemuck::bytes_of(&perturb_gpu),
                );
            }
        }

        // Upload roots if needed
        if params.fractal_type.needs_roots() {
            let roots = params.compute_roots();
            let roots_flat: Vec<f32> = roots.iter().flat_map(|r| r.iter().copied()).collect();
            let mut padded = roots_flat;
            padded.resize(32, 0.0); // 16 roots * 2 floats
            self.queue
                .write_buffer(&self.roots_buffer, 0, bytemuck::cast_slice(&padded));
        }

        let mut samples = crate::fractals::compute_samples(ss);
        // In median mode, clamp to MAX_MEDIAN_SAMPLES (iterations buffer capacity)
        if use_median && samples.len() > MAX_MEDIAN_SAMPLES as usize {
            samples.truncate(MAX_MEDIAN_SAMPLES as usize);
        }
        let num_samples = samples.len() as u32;
        let params_size = std::mem::size_of::<GpuParams>() as u64;

        // Stage all sample params into staging buffer (CPU side, no GPU sync needed)
        let base_gpu_params = params.to_gpu_params(self.display_width, self.height, self.width);
        for (i, &(offset_x, offset_y, weight)) in samples.iter().enumerate() {
            let mut gpu_params = base_gpu_params;
            gpu_params.sample_offset = [offset_x, offset_y];
            gpu_params.sample_weight = weight;
            gpu_params.sample_index = if use_median { i as u32 } else { 0 };
            gpu_params.num_samples = num_samples;
            if use_perturb {
                gpu_params.pixel_step = [ps_mantissa_x, ps_mantissa_y];
            }
            self.queue.write_buffer(
                &self.params_staging_buffer,
                i as u64 * params_size,
                bytemuck::bytes_of(&gpu_params),
            );
        }

        // Single command buffer for ALL work: clear + N×(copy params, iterate, colorize) + finalize + readback
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render"),
            });

        if !use_median {
            encoder.clear_buffer(&self.accum_buffer, 0, None);
        }

        for i in 0..samples.len() {
            // Copy this sample's params from staging to the uniform buffer
            encoder.copy_buffer_to_buffer(
                &self.params_staging_buffer,
                i as u64 * params_size,
                &self.params_buffer,
                0,
                params_size,
            );

            // Fractal iteration
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("iterate"),
                    timestamp_writes: None,
                });
                if use_perturb {
                    pass.set_pipeline(&self.perturb_pipeline);
                    pass.set_bind_group(0, &self.perturb_bind_group, &[]);
                } else if params.fractal_type.is_escape_time() {
                    pass.set_pipeline(&self.escape_pipeline);
                    pass.set_bind_group(0, &self.escape_bind_group, &[]);
                } else {
                    pass.set_pipeline(&self.newton_pipeline);
                    pass.set_bind_group(0, &self.newton_bind_group, &[]);
                }
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            // Colorize and accumulate (only in non-median mode)
            if !use_median {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("accumulate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.colorize_pipeline);
                pass.set_bind_group(0, &self.colorize_bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }

        // Finalize pass
        {
            let mut fin_params = base_gpu_params;
            fin_params.num_samples = num_samples;
            if use_perturb {
                fin_params.pixel_step = [ps_mantissa_x, ps_mantissa_y];
            }
            if !use_median {
                let total_weight: f32 = samples.iter().map(|s| s.2).sum();
                fin_params.sample_weight = total_weight;
            }
            let fin_offset = samples.len() as u64 * params_size;
            self.queue.write_buffer(&self.params_staging_buffer, fin_offset, bytemuck::bytes_of(&fin_params));
            encoder.copy_buffer_to_buffer(
                &self.params_staging_buffer, fin_offset,
                &self.params_buffer, 0,
                params_size,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("finalize"),
                timestamp_writes: None,
            });
            if use_median {
                pass.set_pipeline(&self.median_finalize_pipeline);
                pass.set_bind_group(0, &self.median_finalize_bind_group, &[]);
            } else {
                pass.set_pipeline(&self.finalize_pipeline);
                pass.set_bind_group(0, &self.finalize_bind_group, &[]);
            }
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy output buffer -> readback buffer
        let buf_size = (self.width as u64) * (self.height as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.output_buffer, 0,
            &self.readback_buffer, 0,
            buf_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        self.last_render_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    /// Read pixels back from the GPU readback buffer.
    /// Returns RGBA bytes at display_width × height (padding stripped).
    pub fn read_pixels(&self) -> Vec<u8> {
        let slice = self.readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let stride = self.width;
        let dw = self.display_width;

        let pixels = if stride == dw {
            data[..(dw * self.height * 4) as usize].to_vec()
        } else {
            let mut out = Vec::with_capacity((dw * self.height * 4) as usize);
            for row in 0..self.height {
                let start = (row * stride * 4) as usize;
                let end = start + (dw * 4) as usize;
                out.extend_from_slice(&data[start..end]);
            }
            out
        };
        drop(data);
        self.readback_buffer.unmap();
        pixels
    }

    /// Export the current fractal to a PNG file.
    pub fn export_png(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let pixels = self.read_pixels();
        let img = image::RgbaImage::from_raw(self.display_width, self.height, pixels)
            .ok_or("Failed to create image")?;
        img.save(path)?;
        Ok(())
    }
}

// -- Helpers ------------------------------------------------------------------

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

