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
    /// Perturbation reference point (view center, or a minibrot nucleus).
    ref_re: Float,
    ref_im: Float,
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
        let mut ref_re = params.center_re.clone();
        let mut ref_im = params.center_im.clone();
        let (orbit_len, bla_num_levels) = if is_mandelbrot {
            // Search for a nucleus within at most 1e4 px (at the deepest step)
            // so the reference offset stays exactly representable in the
            // shader's double-single pixel units, even for zoom videos whose
            // delta_c_max spans the shallowest frame.
            let search_r = delta_c_max.min(pixel_step * 1e4);
            let max_period = params.max_iter.min(fractals::BLA_MAX_REF_LEN as u32);
            let reference = fractals::find_mandelbrot_reference(
                &params.center_re, &params.center_im, search_r, max_period, pixel_step,
            );
            let dist = Float::with_val(128, &reference.re - &params.center_re).to_f64()
                .hypot(Float::with_val(128, &reference.im - &params.center_im).to_f64());
            let perturb_data = fractals::compute_mandelbrot_with_bla(
                &reference.re, &reference.im,
                params.max_iter, pixel_step,
                (delta_c_max + dist).max(1e-300), fractals::BLA_EPS, reference.period,
            );
            if let Some(p) = reference.period {
                status(format!("Reference: period-{p} nucleus, {:.0} px from center", dist / pixel_step));
            }
            ref_re = reference.re;
            ref_im = reference.im;
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
                params.fractal_type, julia_c, params.power,
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
            ref_re,
            ref_im,
        });
        Ok(())
    }

    /// Dispatch `pipe` over the frame in horizontal bands, one submit each,
    /// with band height adapted to ~0.75 s of GPU time (`band_wg` carries the
    /// estimate between calls). A single whole-frame dispatch can run for
    /// minutes (deep zoom, high max_iter, no BLA, median neighbor palettes)
    /// and trips the driver watchdog (NVIDIA Xid 109), which silently leaves
    /// the buffers zeroed. Shaders offset their row by params.row_offset.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_banded(
        &self,
        pipe: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        base: GpuParams,
        wg_x: u32,
        wg_y: u32,
        band_wg: &mut u32,
        label: &str,
        status: &dyn Fn(String),
    ) {
        const BAND_TARGET: f64 = 0.75;
        let t_start = std::time::Instant::now();
        let mut last_status = std::time::Instant::now();
        let mut wg_row = 0u32;
        while wg_row < wg_y {
            let n_wg = (*band_wg).clamp(1, wg_y - wg_row);
            let mut gpu_params = base;
            gpu_params.row_offset = wg_row * 16;
            self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&gpu_params));
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, n_wg, 1);
            }
            let t = std::time::Instant::now();
            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            let dt = t.elapsed().as_secs_f64().max(1e-4);
            let scaled = (n_wg as f64 * BAND_TARGET / dt).clamp(1.0, n_wg as f64 * 2.0);
            *band_wg = (scaled as u32).clamp(1, wg_y);
            wg_row += n_wg;
            if last_status.elapsed().as_secs_f64() > 2.0 {
                let frac = wg_row as f64 / wg_y as f64;
                let el = t_start.elapsed().as_secs_f64();
                status(format!(
                    "{label}: {:.0}% ({:.0}s elapsed, ~{:.0}s left)",
                    frac * 100.0, el, el / frac.max(1e-6) * (1.0 - frac)
                ));
                last_status = std::time::Instant::now();
            }
        }
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
        // Neighbor-sampling palettes can't be multi-sample *averaged*, but the
        // median path works: their neighbor reads go through iter_at(), which
        // returns the per-pixel median field.
        let ss = if params.palette.uses_neighbor_sampling() && !use_median { 1 } else { params.supersampling };
        let samples = if use_median {
            fractals::compute_samples_median(ss)
        } else {
            fractals::compute_samples(ss)
        };

        let pixel_step = params.pixel_step_x(self.display_w);
        let use_perturb = params.fractal_type.is_escape_time()
            && (params.fractal_type != FractalType::Multibrot
                || (params.power - params.power.round()).abs() < 1e-4)
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
            // Orbit traps / stripes / interior fields only sample on
            // single-step iterations; keep those palettes per-step so BLA
            // jumps don't starve their accumulators.
            let bla_num_levels = if params.palette.needs_per_step_orbit() || params.no_bla {
                0
            } else {
                cache.bla_num_levels
            };
            let pgpu = PerturbGpuParams {
                ref_orbit_len: cache.orbit_len,
                pixel_step_exp: ps_exp,
                bla_num_levels,
                _pad: 0,
                ref_off: PerturbGpuParams::ref_offset(params, &cache.ref_re, &cache.ref_im, pixel_step, step_y),
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
        let base_gpu_params = params.to_gpu_params(self.display_w, self.height, self.stride);
        let sample_params = |i: usize| {
            let (offset_x, offset_y, weight) = samples[i];
            let mut gpu_params = base_gpu_params;
            gpu_params.sample_offset = [offset_x, offset_y];
            gpu_params.sample_weight = weight;
            gpu_params.sample_index = if use_median { i as u32 } else { 0 };
            gpu_params.num_samples = num_samples;
            if use_perturb {
                gpu_params.pixel_step = [ps_mantissa_x, ps_mantissa_y];
            }
            gpu_params
        };

        let wg_x = self.display_w.div_ceil(16);
        let wg_y = self.height.div_ceil(16);

        if !use_median {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.clear_buffer(&self.accum_buf, 0, None);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        let n_samp = samples.len();
        let iter_pipe = if use_perturb {
            (&self.perturb_pipe, &self.perturb_bg)
        } else if params.fractal_type.is_escape_time() {
            (&self.esc_pipe, &self.esc_bg)
        } else {
            (&self.new_pipe, &self.new_bg)
        };
        let mut band_wg: u32 = 4;
        for i in 0..n_samp {
            let label = if n_samp > 1 { format!("Iterating sample {}/{}", i + 1, n_samp) } else { "Iterating".to_string() };
            self.dispatch_banded(iter_pipe.0, iter_pipe.1, sample_params(i), wg_x, wg_y, &mut band_wg, &label, status);
            if !use_median {
                let gpu_params = sample_params(i);
                self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&gpu_params));
                let mut encoder = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.col_pipe);
                    pass.set_bind_group(0, &self.col_bg, &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
                self.queue.submit(std::iter::once(encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait);
            }
        }

        let mut fin_params = base_gpu_params;
        fin_params.num_samples = num_samples;
        if !use_median {
            let total_weight: f32 = samples.iter().map(|s| s.2).sum();
            fin_params.sample_weight = total_weight;
        }
        if use_perturb {
            fin_params.pixel_step = [ps_mantissa_x, ps_mantissa_y];
        }
        // Median finalize with a neighbor-reading palette takes a per-sample
        // median for every neighbor read, so it can be as slow as iterating:
        // band it too.
        let (fin_pipe, fin_bg) = if use_median { (&self.med_pipe, &self.med_bg) } else { (&self.fin_pipe, &self.fin_bg) };
        let mut fin_band_wg: u32 = 4;
        self.dispatch_banded(fin_pipe, fin_bg, fin_params, wg_x, wg_y, &mut fin_band_wg, "Coloring", status);
        let mut encoder = self.device.create_command_encoder(&Default::default());
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
        // finalize writes alpha 255 everywhere; an all-zero buffer means the
        // GPU work never ran (device lost / watchdog), not a black image.
        if data.iter().all(|&b| b == 0) {
            drop(data);
            self.readback_buf.unmap();
            return Err("GPU produced no output (device lost or driver watchdog timeout?)".into());
        }
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

/// Overlap above and below each strip. Neighbor-reading palettes look at most
/// ~6 px away (biolum: 4.5 px kernel + 1 px gradient, x ref scale), so 16 px
/// keeps every kept row's neighborhood inside its own strip.
const STRIP_MARGIN: u32 = 16;

/// Render the frame, splitting it into horizontal strips when its buffers
/// would exceed the GPU's 2 GiB per-buffer limit (e.g. 9-sample median at
/// poster sizes). Each strip is its own view whose center and half-height are
/// derived from the full frame's pixel step, so strip pixels sample exactly
/// the full frame's points; only each strip's interior rows are kept.
/// (Palettes that seed noise from the view center, e.g. Storm, would show
/// seams; the neighbor kernels themselves are seamless.)
/// FRACTAL_MAX_STRIP_ROWS overrides the row budget, for testing strip seams.
fn render_in_strips(
    params: &FractalParams,
    width: u32,
    height: u32,
    status: &dyn Fn(String),
) -> Result<Vec<u8>, String> {
    let slots = if params.use_median { MAX_MEDIAN_SAMPLES as u64 } else { 1 };
    let stride = crate::gpu::align_width(width) as u64;
    // Largest per-pixel buffer: iterations (4 B x slots) or a vec4<f32> buffer (16 B).
    let bytes_per_row = stride * (4 * slots).max(16);
    let mut max_rows = ((1900u64 << 20) / bytes_per_row).min(u32::MAX as u64) as u32;
    if let Some(r) = std::env::var("FRACTAL_MAX_STRIP_ROWS").ok().and_then(|v| v.parse::<u32>().ok()) {
        max_rows = max_rows.min(r.max(2 * STRIP_MARGIN + 16));
    }
    if height <= max_rows {
        let mut ctx = ExportContext::new(width, height, params.max_iter, params.use_median)?;
        return ctx.render_frame(params, status);
    }

    let core_max = max_rows - 2 * STRIP_MARGIN;
    let n = height.div_ceil(core_max);
    let core = height.div_ceil(n);
    let hs = (core + 2 * STRIP_MARGIN).min(height);
    let step_y = params.pixel_step_y(height);
    let mut ctx = ExportContext::new(width, hs, params.max_iter, params.use_median)?;
    let row_bytes = (width * 4) as usize;
    let mut out = vec![0u8; row_bytes * height as usize];
    for s in 0..n {
        let r0 = s * core;
        let r1 = ((s + 1) * core).min(height);
        let top = r0.saturating_sub(STRIP_MARGIN).min(height - hs);
        let mut sp = params.clone();
        let off_rows = (top as f64 + (hs - 1) as f64 / 2.0) - (height - 1) as f64 / 2.0;
        // Rows run top to bottom = decreasing imaginary part.
        sp.center_im -= off_rows * step_y;
        sp.half_range_y = step_y * (hs - 1) as f64 / 2.0;
        status(format!("Strip {}/{} (rows {}..{})", s + 1, n, r0, r1));
        let strip = ctx.render_frame(&sp, &|msg| status(format!("Strip {}/{}: {msg}", s + 1, n)))?;
        for r in r0..r1 {
            let src = (r - top) as usize * row_bytes;
            let dst = r as usize * row_bytes;
            out[dst..dst + row_bytes].copy_from_slice(&strip[src..src + row_bytes]);
        }
    }
    Ok(out)
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

    let pixels = render_in_strips(&params, config.width, config.height, &|msg| status_callback(msg))?;

    let img = image::RgbaImage::from_raw(config.width, config.height, pixels)
        .ok_or_else(|| "Failed to create image from pixel data".to_string())?;

    status_callback("Saving PNG...".to_string());
    img.save(&path).map_err(|e| format!("Save failed: {e}"))?;

    let ss = if params.palette.uses_neighbor_sampling() && !params.use_median { 1 } else { params.supersampling };
    let ss_info = if ss > 1 { " (supersampled)".to_string() } else { String::new() };
    Ok(format!("{}x{}{} -> {}", config.width, config.height, ss_info, path))
}
