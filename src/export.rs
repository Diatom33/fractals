//! Headless export pipeline: single high-res PNGs and zoom-video frame
//! sequences. Creates its own wgpu device/queue, independent of the app's GPU
//! state. `ExportContext` owns the device, pipelines, and buffers so many
//! frames can be rendered without redoing setup, and caches the perturbation
//! reference orbit + BLA tree across frames.

use crate::fractals::{self, BlaCoeff, FractalParams, FractalType, GpuParams, PerturbGpuParams};
use crate::gpu::MAX_MEDIAN_SAMPLES;
use rug::Float;

pub struct ExportConfig {
    pub width: u32,
    pub height: u32,
    pub ss: u32,
    pub max_iter: Option<u32>,
    pub path: String,
}

pub fn expand_tilde(path: &str) -> String {
    if path == "~" || path.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{}{}", home, &path[1..]);
        }
    }
    path.to_string()
}

fn align_width(w: u32) -> u32 {
    w.div_ceil(64) * 64
}

/// What the uploaded reference orbit + BLA tree were computed for.
/// BLA validity radii shrink as delta_c_max grows and orbit precision improves
/// as pixel_step shrinks, so a cached orbit stays valid for any frame with
/// `delta_c_max <= cached.delta_c_max && pixel_step >= cached.pixel_step` —
/// which is every later frame of a zoom-in video prepared at full depth.
struct OrbitCache {
    center_re: Float,
    center_im: Float,
    max_iter: u32,
    fractal_type: u32,
    julia_c: [f32; 2],
    orbit_len: u32,
    bla_num_levels: u32,
    delta_c_max: f64,
    pixel_step: f64,
}

/// A headless wgpu device plus all pipelines and buffers needed to render
/// frames at a fixed resolution / AA mode / max-iteration budget.
pub struct ExportContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    display_w: u32,
    stride: u32,
    height: u32,
    max_iter_cap: u32,
    use_median: bool,

    esc_pipe: wgpu::ComputePipeline,
    esc_bg: wgpu::BindGroup,
    perturb_pipe: wgpu::ComputePipeline,
    perturb_bg: wgpu::BindGroup,
    new_pipe: wgpu::ComputePipeline,
    new_bg: wgpu::BindGroup,
    col_pipe: wgpu::ComputePipeline,
    col_bg: wgpu::BindGroup,
    fin_pipe: wgpu::ComputePipeline,
    fin_bg: wgpu::BindGroup,
    med_pipe: wgpu::ComputePipeline,
    med_bg: wgpu::BindGroup,

    params_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    accum_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    roots_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    ref_orbit_buf: wgpu::Buffer,
    perturb_params_buf: wgpu::Buffer,
    bla_buf: wgpu::Buffer,

    orbit_cache: Option<OrbitCache>,
}

impl ExportContext {
    /// `max_iter` bounds the reference-orbit/BLA buffer sizes; frames may use
    /// any max_iter up to it. `use_median` picks the AA pipeline the iteration
    /// buffer is sized for.
    pub fn new(width: u32, height: u32, max_iter: u32, use_median: bool) -> Result<Self, String> {
        let display_w = width;
        let stride = align_width(width);
        let height_ = height;
        let out_pixels = (stride as u64) * (height as u64);
        let iter_slots = if use_median { MAX_MEDIAN_SAMPLES as u64 } else { 1 };

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok_or_else(|| "No GPU adapter found".to_string())?;

        // Match the GUI's raised buffer limits (main.rs) up to what the adapter
        // supports — default limits cap storage bindings at 128 MiB, which
        // deep-zoom BLA trees and median iteration buffers exceed.
        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size.min(1 << 31),
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("export"),
                required_limits: limits.clone(),
                ..Default::default()
            },
            None,
        ))
        .map_err(|e| format!("Failed to create device: {e}"))?;

        // Median mode holds all sub-pixel samples' iteration counts at once;
        // refuse clearly rather than dying in wgpu validation at large sizes.
        let iter_bytes = out_pixels * 4 * iter_slots;
        if iter_bytes > limits.max_storage_buffer_binding_size as u64 {
            return Err(format!(
                "{}x{} with median AA needs a {} MiB iterations buffer (GPU limit {} MiB). \
                 Use the Accumulate AA filter or a smaller resolution.",
                display_w, height,
                iter_bytes >> 20,
                limits.max_storage_buffer_binding_size >> 20,
            ));
        }

        let shader = |src: String| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let escape_shader = shader(include_str!("shaders/escape.wgsl").into());
        let newton_shader = shader(include_str!("shaders/newton.wgsl").into());
        let colorize_shader = shader(crate::shaders::colorize());
        let finalize_shader = shader(include_str!("shaders/finalize.wgsl").into());
        let median_finalize_shader = shader(crate::shaders::median_finalize());
        let perturb_shader = shader(include_str!("shaders/escape_perturb.wgsl").into());

        let storage_buf = |size: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let none = wgpu::BufferUsages::empty();

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let iter_buf = storage_buf(iter_bytes, none);
        let z_buf = storage_buf(out_pixels * 16, none);
        let orbit_trap_buf = storage_buf(out_pixels * 16, none);
        let accum_buf = storage_buf(out_pixels * 16, wgpu::BufferUsages::COPY_DST);
        let out_buf = storage_buf(out_pixels * 4, wgpu::BufferUsages::COPY_SRC);
        let roots_buf = storage_buf(8 * 16, wgpu::BufferUsages::COPY_DST);
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: out_pixels * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });

        let ref_orbit_buf = storage_buf((max_iter as u64 + 1) * 16, wgpu::BufferUsages::COPY_DST);
        let perturb_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: std::mem::size_of::<PerturbGpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        // BLA tree (Mandelbrot only). Above BLA_MAX_REF_LEN the tree isn't
        // built at all (compute_mandelbrot_with_bla returns it empty), so cap
        // the allocation to match; min 64 bytes for the non-perturb fallback.
        let bla_cap = (max_iter as u64 + 1).min(fractals::BLA_MAX_REF_LEN as u64 + 1);
        let bla_num_levels_max = ((bla_cap as f64).log2().ceil() as u32) + 2;
        let bla_buf_size = bla_cap * bla_num_levels_max as u64 * std::mem::size_of::<BlaCoeff>() as u64;
        let bla_buf = storage_buf(bla_buf_size.max(64), wgpu::BufferUsages::COPY_DST);

        // Staging for per-sample GpuParams (64 accumulate samples max, +1 finalize)
        let params_size = std::mem::size_of::<GpuParams>() as u64;
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 65 * params_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        let bgl_uniform = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl_storage = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        macro_rules! be {
            ($b:expr, $buf:expr) => {
                wgpu::BindGroupEntry { binding: $b, resource: $buf.as_entire_binding() }
            };
        }
        let pipeline = |module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None, bind_group_layouts: &[layout], push_constant_ranges: &[],
                })),
                module, entry_point: Some("main"),
                compilation_options: Default::default(), cache: None,
            })
        };

        // Escape (params, iterations, final_z, orbit_traps)
        let esc_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_uniform(0), bgl_storage(1, false), bgl_storage(2, false), bgl_storage(3, false)],
        });
        let esc_pipe = pipeline(&escape_shader, &esc_layout);
        let esc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &esc_layout,
            entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &orbit_trap_buf)],
        });

        // Perturbation (params, iterations, final_z, ref_orbit, perturb_params, orbit_traps, bla_tree)
        let perturb_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_uniform(0), bgl_storage(1, false), bgl_storage(2, false), bgl_storage(3, true), bgl_uniform(4), bgl_storage(5, false), bgl_storage(6, true)],
        });
        let perturb_pipe = pipeline(&perturb_shader, &perturb_layout);
        let perturb_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &perturb_layout,
            entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &ref_orbit_buf), be!(4, &perturb_params_buf), be!(5, &orbit_trap_buf), be!(6, &bla_buf)],
        });

        // Newton (params, iterations, final_z, roots, orbit_traps)
        let new_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_uniform(0), bgl_storage(1, false), bgl_storage(2, false), bgl_storage(3, true), bgl_storage(4, false)],
        });
        let new_pipe = pipeline(&newton_shader, &new_layout);
        let new_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &new_layout,
            entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &roots_buf), be!(4, &orbit_trap_buf)],
        });

        // Colorize (params, iterations[ro], final_z[ro], accum[rw], roots[ro], orbit_traps[ro])
        let col_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_uniform(0), bgl_storage(1, true), bgl_storage(2, true), bgl_storage(3, false), bgl_storage(4, true), bgl_storage(5, true)],
        });
        let col_pipe = pipeline(&colorize_shader, &col_layout);
        let col_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &col_layout,
            entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &accum_buf), be!(4, &roots_buf), be!(5, &orbit_trap_buf)],
        });

        // Finalize (params, accum[ro], output[rw])
        let fin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_uniform(0), bgl_storage(1, true), bgl_storage(2, false)],
        });
        let fin_pipe = pipeline(&finalize_shader, &fin_layout);
        let fin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &fin_layout,
            entries: &[be!(0, &params_buf), be!(1, &accum_buf), be!(2, &out_buf)],
        });

        // Median finalize (params, iterations[ro], output[rw], final_z[ro], roots[ro], orbit_traps[ro])
        let med_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_uniform(0), bgl_storage(1, true), bgl_storage(2, false), bgl_storage(3, true), bgl_storage(4, true), bgl_storage(5, true)],
        });
        let med_pipe = pipeline(&median_finalize_shader, &med_layout);
        let med_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &med_layout,
            entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &out_buf), be!(3, &z_buf), be!(4, &roots_buf), be!(5, &orbit_trap_buf)],
        });

        Ok(Self {
            device,
            queue,
            display_w,
            stride,
            height: height_,
            max_iter_cap: max_iter,
            use_median,
            esc_pipe, esc_bg,
            perturb_pipe, perturb_bg,
            new_pipe, new_bg,
            col_pipe, col_bg,
            fin_pipe, fin_bg,
            med_pipe, med_bg,
            params_buf,
            staging_buf,
            accum_buf,
            out_buf,
            roots_buf,
            readback_buf,
            ref_orbit_buf,
            perturb_params_buf,
            bla_buf,
            orbit_cache: None,
        })
    }

    /// Compute + upload the reference orbit (and BLA tree for Mandelbrot) for
    /// the given precision/validity envelope, unless the cache already covers
    /// it. For zoom videos, call once with the DEEPEST frame's pixel_step and
    /// the shallowest perturbed frame's delta_c_max so every frame reuses it.
    pub fn ensure_orbit(
        &mut self,
        params: &FractalParams,
        pixel_step: f64,
        delta_c_max: f64,
        status: &dyn Fn(String),
    ) -> Result<(), String> {
        let ft_idx = params.fractal_type.shader_index();
        let cache_ok = self.orbit_cache.as_ref().is_some_and(|c| {
            c.center_re == params.center_re
                && c.center_im == params.center_im
                && c.max_iter == params.max_iter
                && c.fractal_type == ft_idx
                && c.julia_c == params.julia_c
                && delta_c_max <= c.delta_c_max
                && pixel_step >= c.pixel_step
        });
        if cache_ok {
            return Ok(());
        }
        if params.max_iter > self.max_iter_cap {
            return Err(format!(
                "max_iter {} exceeds this context's capacity {}",
                params.max_iter, self.max_iter_cap
            ));
        }

        status("Computing reference orbit...".to_string());
        let is_mandelbrot = params.fractal_type == FractalType::Mandelbrot;
        let (orbit_len, bla_num_levels) = if is_mandelbrot {
            let eps = 1.0 / 1024.0;
            let perturb_data = fractals::compute_mandelbrot_with_bla(
                &params.center_re, &params.center_im,
                params.max_iter, pixel_step,
                delta_c_max.max(1e-300), eps,
            );
            self.queue.write_buffer(&self.ref_orbit_buf, 0, bytemuck::cast_slice(&perturb_data.orbit));
            if !perturb_data.bla.is_empty() {
                self.queue.write_buffer(&self.bla_buf, 0, bytemuck::cast_slice(&perturb_data.bla));
            }
            (perturb_data.orbit_len, perturb_data.bla_num_levels)
        } else {
            let julia_c = if params.fractal_type == FractalType::Julia {
                Some((params.julia_c[0] as f64, params.julia_c[1] as f64))
            } else {
                None
            };
            let perturb_data = fractals::compute_variant_reference_orbit(
                &params.center_re, &params.center_im,
                params.max_iter, pixel_step,
                params.fractal_type, julia_c,
            );
            self.queue.write_buffer(&self.ref_orbit_buf, 0, bytemuck::cast_slice(&perturb_data.orbit));
            (perturb_data.orbit_len, 0u32)
        };
        status(format!(
            "Perturbation: {} ref orbit iters{}",
            orbit_len,
            if bla_num_levels > 0 { format!(", BLA {} levels", bla_num_levels) } else { String::new() }
        ));

        self.orbit_cache = Some(OrbitCache {
            center_re: params.center_re.clone(),
            center_im: params.center_im.clone(),
            max_iter: params.max_iter,
            fractal_type: ft_idx,
            julia_c: params.julia_c,
            orbit_len,
            bla_num_levels,
            delta_c_max,
            pixel_step,
        });
        Ok(())
    }

    /// Render one frame and return RGBA pixels at display_w x height
    /// (stride padding stripped). The context's use_median mode wins over
    /// params.use_median (buffers are sized for it).
    pub fn render_frame(
        &mut self,
        params: &FractalParams,
        status: &dyn Fn(String),
    ) -> Result<Vec<u8>, String> {
        if params.max_iter > self.max_iter_cap {
            return Err(format!(
                "max_iter {} exceeds this context's capacity {}",
                params.max_iter, self.max_iter_cap
            ));
        }
        let use_median = self.use_median;
        // Neighbor-sampling palettes can't be multi-sample averaged.
        let ss = if params.palette.uses_neighbor_sampling() { 1 } else { params.supersampling };
        let samples = if use_median {
            fractals::compute_samples_median(ss)
        } else {
            fractals::compute_samples(ss)
        };

        let pixel_step = params.pixel_step_x(self.display_w);
        let use_perturb = params.fractal_type.is_escape_time()
            && params.fractal_type != FractalType::Multibrot
            && pixel_step < 1e-7;

        let step_y = params.pixel_step_y(self.height);
        let ps_exp = pixel_step.log2().floor() as i32;
        let ps_scale = 2.0_f64.powi(ps_exp);
        let ps_mantissa_x = (pixel_step / ps_scale) as f32;
        let ps_mantissa_y = (step_y / ps_scale) as f32;

        if use_perturb {
            let delta_c_max = params.half_range_x.hypot(params.half_range_y).max(1e-300);
            self.ensure_orbit(params, pixel_step, delta_c_max, status)?;
            let cache = self.orbit_cache.as_ref().unwrap();
            // Orbit traps only sample on single-step iterations; keep the
            // Canopy palette per-step so BLA jumps don't starve its highlights.
            let bla_num_levels = if params.palette == fractals::ColorPalette::Canopy {
                0
            } else {
                cache.bla_num_levels
            };
            let pgpu = PerturbGpuParams {
                ref_orbit_len: cache.orbit_len,
                pixel_step_exp: ps_exp,
                bla_num_levels,
                _pad: 0,
            };
            self.queue.write_buffer(&self.perturb_params_buf, 0, bytemuck::bytes_of(&pgpu));
        }

        if params.fractal_type.needs_roots() {
            let roots = params.compute_roots();
            let mut flat: Vec<f32> = roots.iter().flat_map(|r| r.iter().copied()).collect();
            flat.resize(32, 0.0);
            self.queue.write_buffer(&self.roots_buf, 0, bytemuck::cast_slice(&flat));
        }

        let num_samples = samples.len() as u32;
        let params_size = std::mem::size_of::<GpuParams>() as u64;
        let base_gpu_params = params.to_gpu_params(self.display_w, self.height, self.stride);
        for (i, &(offset_x, offset_y, weight)) in samples.iter().enumerate() {
            let mut gpu_params = base_gpu_params;
            gpu_params.sample_offset = [offset_x, offset_y];
            gpu_params.sample_weight = weight;
            gpu_params.sample_index = if use_median { i as u32 } else { 0 };
            gpu_params.num_samples = num_samples;
            if use_perturb {
                gpu_params.pixel_step = [ps_mantissa_x, ps_mantissa_y];
            }
            self.queue.write_buffer(&self.staging_buf, i as u64 * params_size, bytemuck::bytes_of(&gpu_params));
        }

        let wg_x = self.display_w.div_ceil(16);
        let wg_y = self.height.div_ceil(16);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        if !use_median {
            encoder.clear_buffer(&self.accum_buf, 0, None);
        }

        for i in 0..samples.len() {
            encoder.copy_buffer_to_buffer(&self.staging_buf, i as u64 * params_size, &self.params_buf, 0, params_size);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                if use_perturb {
                    pass.set_pipeline(&self.perturb_pipe);
                    pass.set_bind_group(0, &self.perturb_bg, &[]);
                } else if params.fractal_type.is_escape_time() {
                    pass.set_pipeline(&self.esc_pipe);
                    pass.set_bind_group(0, &self.esc_bg, &[]);
                } else {
                    pass.set_pipeline(&self.new_pipe);
                    pass.set_bind_group(0, &self.new_bg, &[]);
                }
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            if !use_median {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.col_pipe);
                pass.set_bind_group(0, &self.col_bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }

        {
            let mut fin_params = base_gpu_params;
            fin_params.num_samples = num_samples;
            if !use_median {
                let total_weight: f32 = samples.iter().map(|s| s.2).sum();
                fin_params.sample_weight = total_weight;
            }
            if use_perturb {
                fin_params.pixel_step = [ps_mantissa_x, ps_mantissa_y];
            }
            let fin_offset = samples.len() as u64 * params_size;
            self.queue.write_buffer(&self.staging_buf, fin_offset, bytemuck::bytes_of(&fin_params));
            encoder.copy_buffer_to_buffer(&self.staging_buf, fin_offset, &self.params_buf, 0, params_size);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            if use_median {
                pass.set_pipeline(&self.med_pipe);
                pass.set_bind_group(0, &self.med_bg, &[]);
            } else {
                pass.set_pipeline(&self.fin_pipe);
                pass.set_bind_group(0, &self.fin_bg, &[]);
            }
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        let out_bytes = (self.stride as u64) * (self.height as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.readback_buf, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let slice = self.readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|e| format!("Map recv failed: {e}"))?.map_err(|e| format!("Map failed: {e}"))?;

        let data = slice.get_mapped_range();
        // Strip stride padding columns
        let pixels = if self.stride == self.display_w {
            data[..(self.display_w * self.height * 4) as usize].to_vec()
        } else {
            let mut out = Vec::with_capacity((self.display_w * self.height * 4) as usize);
            for row in 0..self.height {
                let start = (row * self.stride * 4) as usize;
                let end = start + (self.display_w * 4) as usize;
                out.extend_from_slice(&data[start..end]);
            }
            out
        };
        drop(data);
        self.readback_buf.unmap();
        Ok(pixels)
    }
}

/// Aspect-correct params for the export dimensions: keep pixels square by
/// expanding the smaller axis (never cropping), same policy as the interactive
/// view. Without this, exporting at an aspect ratio different from the current
/// view stretches the image.
pub fn aspect_correct(params: &mut FractalParams, width: u32, height: u32) {
    if width > 0 && height > 0 {
        let scale_x = 2.0 * params.half_range_x / width as f64;
        let scale_y = 2.0 * params.half_range_y / height as f64;
        let scale = scale_x.max(scale_y);
        params.half_range_x = scale * width as f64 / 2.0;
        params.half_range_y = scale * height as f64 / 2.0;
    }
}

pub fn export_headless(
    params: &FractalParams,
    config: &ExportConfig,
    status_callback: impl Fn(String) + Send,
) -> Result<String, String> {
    let mut params = params.clone();
    if let Some(max_iter) = config.max_iter {
        params.max_iter = max_iter;
    }
    params.supersampling = config.ss;
    aspect_correct(&mut params, config.width, config.height);

    let path = expand_tilde(&config.path);
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create directory {}: {e}", parent.display()))?;
    }

    status_callback(format!(
        "Exporting {} at {}x{} ...",
        params.fractal_type.name(), config.width, config.height
    ));

    let mut ctx = ExportContext::new(config.width, config.height, params.max_iter, params.use_median)?;
    let pixels = ctx.render_frame(&params, &|msg| status_callback(msg))?;

    let img = image::RgbaImage::from_raw(config.width, config.height, pixels)
        .ok_or_else(|| "Failed to create image from pixel data".to_string())?;

    status_callback("Saving PNG...".to_string());
    img.save(&path).map_err(|e| format!("Save failed: {e}"))?;

    let ss = if params.palette.uses_neighbor_sampling() { 1 } else { params.supersampling };
    let ss_info = if ss > 1 { " (supersampled)".to_string() } else { String::new() };
    Ok(format!("{}x{}{} -> {}", config.width, config.height, ss_info, path))
}
