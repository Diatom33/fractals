mod app;
mod export;
mod fractals;
mod gpu;
mod nebula;

fn main() -> eframe::Result {
    env_logger::init();

    // Quick CLI export mode: --export <file.png> [--type mandelbrot|julia|etc]
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--export") {
        if let Some(path) = args.get(pos + 1) {
            if is_nebulabrot(&args) {
                return export_nebulabrot(&args, path);
            }
            return export_cli(&args, path);
        }
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 900.0])
            .with_min_inner_size([640.0, 480.0])
            .with_title("Fractal Explorer"),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            present_mode: wgpu::PresentMode::AutoNoVsync,
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
                eframe::egui_wgpu::WgpuSetupCreateNew {
                    device_descriptor: std::sync::Arc::new(|adapter| {
                        let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                            wgpu::Limits::downlevel_webgl2_defaults()
                        } else {
                            wgpu::Limits::default()
                        };
                        wgpu::DeviceDescriptor {
                            label: Some("egui wgpu device"),
                            required_features: wgpu::Features::default(),
                            required_limits: wgpu::Limits {
                                max_texture_dimension_2d: 8192,
                                max_buffer_size: 1 << 30, // 1GB
                                max_storage_buffer_binding_size: 1 << 30,
                                ..base_limits
                            },
                            memory_hints: wgpu::MemoryHints::default(),
                        }
                    }),
                    ..Default::default()
                },
            ),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "Fractal Explorer",
        options,
        Box::new(|cc| Ok(Box::new(app::FractalApp::new(cc)))),
    )
}

fn is_nebulabrot(args: &[String]) -> bool {
    if let Some(pos) = args.iter().position(|a| a == "--type") {
        if let Some(name) = args.get(pos + 1) {
            let n = name.to_lowercase();
            return n == "nebulabrot" || n == "nebula";
        }
    }
    false
}

fn default_nebula_view(width: u32, height: u32) -> ([f32; 2], [f32; 2]) {
    let center_x = -0.5f32;
    let center_y = 0.0f32;
    let half_y = 1.5f32;
    let aspect = width as f32 / height as f32;
    let half_x = half_y * aspect;
    ([center_x - half_x, center_y - half_y], [center_x + half_x, center_y + half_y])
}

fn export_nebulabrot(args: &[String], path: &str) -> eframe::Result {
    use fractals::{NebulaGpuParams, NebulaFinParams};
    use std::io::Write;

    let width = args.iter().position(|a| a == "--width")
        .and_then(|p| args.get(p + 1))
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1920);
    let height = args.iter().position(|a| a == "--height")
        .and_then(|p| args.get(p + 1))
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1080);
    let align = |w: u32| -> u32 { (w + 63) / 64 * 64 };
    let width = align(width);

    let total_samples: u64 = args.iter().position(|a| a == "--nebula-samples")
        .and_then(|p| args.get(p + 1))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(100_000_000);

    let (max_iter_r, max_iter_g, max_iter_b) = if let Some(pos) = args.iter().position(|a| a == "--nebula-iters") {
        if let Some(val) = args.get(pos + 1) {
            let parts: Vec<u32> = val.split(',').filter_map(|s| s.parse().ok()).collect();
            if parts.len() == 3 { (parts[0], parts[1], parts[2]) }
            else { (5000, 500, 50) }
        } else { (5000, 500, 50) }
    } else { (5000, 500, 50) };

    let (view_min, view_max) = if let Some(pos) = args.iter().position(|a| a == "--bounds") {
        if let Some(val) = args.get(pos + 1) {
            let parts: Vec<f64> = val.split(',').filter_map(|s| s.parse().ok()).collect();
            if parts.len() == 4 { ([parts[0] as f32, parts[2] as f32], [parts[1] as f32, parts[3] as f32]) }
            else { default_nebula_view(width, height) }
        } else { default_nebula_view(width, height) }
    } else { default_nebula_view(width, height) };

    let sample_min = [-2.5f32, -1.5f32];
    let sample_max = [1.5f32, 1.5f32];

    println!("Nebulabrot export: {}x{} -> {}", width, height, path);
    println!("  Samples: {}, Iters: R={}, G={}, B={}", total_samples, max_iter_r, max_iter_g, max_iter_b);
    println!("  View: [{}, {}] x [{}, {}]", view_min[0], view_max[0], view_min[1], view_max[1]);

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("No GPU adapter found");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor { label: Some("nebulabrot export"), ..Default::default() },
        None,
    )).expect("Failed to create device");

    let out_pixels = (width as u64) * (height as u64);
    let hist_size = out_pixels * 4;

    let mk_buf = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label), size, usage, mapped_at_creation: false,
    });
    let stor_rw_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let hist_r_buf = mk_buf("histogram_r", hist_size, stor_rw_dst);
    let hist_g_buf = mk_buf("histogram_g", hist_size, stor_rw_dst);
    let hist_b_buf = mk_buf("histogram_b", hist_size, stor_rw_dst);
    let output_buf = mk_buf("output", out_pixels * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let readback_buf = mk_buf("readback", out_pixels * 4, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
    let hist_readback_buf = mk_buf("hist_readback", hist_size, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);

    let sample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nebula_sample"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nebula_sample.wgsl").into()),
    });
    let finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nebula_finalize"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nebula_finalize.wgsl").into()),
    });

    let nebula_params_buf = mk_buf("nebula_params", std::mem::size_of::<NebulaGpuParams>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
    let fin_params_buf = mk_buf("nebula_fin_params", std::mem::size_of::<NebulaFinParams>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

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
        ($b:expr, $buf:expr) => { wgpu::BindGroupEntry { binding: $b, resource: $buf.as_entire_binding() } };
    }

    let sample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None, entries: &[bgl_uniform(0), bgl_storage(1, false), bgl_storage(2, false), bgl_storage(3, false)],
    });
    let sample_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nebula_sample"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&sample_layout], push_constant_ranges: &[],
        })),
        module: &sample_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    let sample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &sample_layout,
        entries: &[be!(0, &nebula_params_buf), be!(1, &hist_r_buf), be!(2, &hist_g_buf), be!(3, &hist_b_buf)],
    });

    let fin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None, entries: &[bgl_uniform(0), bgl_storage(1, true), bgl_storage(2, true), bgl_storage(3, true), bgl_storage(4, false)],
    });
    let fin_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nebula_finalize"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&fin_layout], push_constant_ranges: &[],
        })),
        module: &finalize_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    let fin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &fin_layout,
        entries: &[be!(0, &fin_params_buf), be!(1, &hist_r_buf), be!(2, &hist_g_buf), be!(3, &hist_b_buf), be!(4, &output_buf)],
    });

    let workgroup_size = 256u32;
    let num_workgroups = 256u32;
    let threads_per_dispatch = (num_workgroups * workgroup_size) as u64;
    let samples_per_thread = 64u32;
    let samples_per_dispatch = threads_per_dispatch * samples_per_thread as u64;
    let num_dispatches = ((total_samples + samples_per_dispatch - 1) / samples_per_dispatch) as u32;

    let start = std::time::Instant::now();
    let progress_interval = (num_dispatches / 20).max(1);

    {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&hist_r_buf, 0, None);
        encoder.clear_buffer(&hist_g_buf, 0, None);
        encoder.clear_buffer(&hist_b_buf, 0, None);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    for dispatch_idx in 0..num_dispatches {
        let nebula_params = NebulaGpuParams {
            resolution: [width, height], stride: width,
            max_iter_r, max_iter_g, max_iter_b,
            samples_per_thread, dispatch_index: dispatch_idx,
            sample_min, sample_max, view_min, view_max,
        };
        queue.write_buffer(&nebula_params_buf, 0, bytemuck::bytes_of(&nebula_params));

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&sample_pipe);
            pass.set_bind_group(0, &sample_bg, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        if (dispatch_idx + 1) % progress_interval == 0 || dispatch_idx == num_dispatches - 1 {
            device.poll(wgpu::Maintain::Wait);
            let pct = ((dispatch_idx + 1) as f64 / num_dispatches as f64 * 100.0) as u32;
            let elapsed = start.elapsed().as_secs_f64();
            let samples_done = (dispatch_idx as u64 + 1) * samples_per_dispatch;
            let rate = samples_done as f64 / elapsed / 1e6;
            print!("\r  Sampling: {}% ({:.0}M samples/sec)", pct, rate);
            std::io::stdout().flush().ok();
        }
    }
    device.poll(wgpu::Maintain::Wait);
    println!();

    let find_exposure = |hist_buf: &wgpu::Buffer| -> u32 {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(hist_buf, 0, &hist_readback_buf, 0, hist_size);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        let slice = hist_readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let values: &[u32] = bytemuck::cast_slice(&data);
        let mut nonzero: Vec<u32> = values.iter().copied().filter(|&v| v > 0).collect();
        let exposure = if nonzero.is_empty() { 0 } else {
            nonzero.sort_unstable();
            nonzero[((nonzero.len() as f64 * 0.995) as usize).min(nonzero.len() - 1)]
        };
        drop(data);
        hist_readback_buf.unmap();
        exposure
    };

    print!("  Computing exposure...");
    std::io::stdout().flush().ok();
    let max_r = find_exposure(&hist_r_buf);
    let max_g = find_exposure(&hist_g_buf);
    let max_b = find_exposure(&hist_b_buf);
    println!(" R={}, G={}, B={}", max_r, max_g, max_b);

    if max_r == 0 && max_g == 0 && max_b == 0 {
        println!("WARNING: All histograms empty. Try increasing --nebula-samples.");
    }

    let fin_params = NebulaFinParams {
        resolution: [width, height], stride: width,
        max_r, max_g, max_b, _pad: [0; 2],
    };
    queue.write_buffer(&fin_params_buf, 0, bytemuck::bytes_of(&fin_params));

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&fin_pipe);
        pass.set_bind_group(0, &fin_bg, &[]);
        pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, out_pixels * 4);
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let img = image::RgbaImage::from_raw(width, height, data.to_vec()).unwrap();
    img.save(path).unwrap();
    println!("Done: {}x{}, {} samples, {:.1}s -> {}", width, height, total_samples, start.elapsed().as_secs_f64(), path);
    Ok(())
}

fn export_cli(args: &[String], path: &str) -> eframe::Result {
    use fractals::{FractalParams, FractalType, GpuParams, PerturbGpuParams};

    let mut params = FractalParams::default();

    // Parse --type
    if let Some(pos) = args.iter().position(|a| a == "--type") {
        if let Some(name) = args.get(pos + 1) {
            for &ft in FractalType::ALL {
                let normalized_ft = ft.name().to_lowercase().replace(' ', "");
                let normalized_input = name.to_lowercase().replace([' ', '-', '_'], "");
                if normalized_ft == normalized_input
                {
                    params.fractal_type = ft;
                    params.set_from_default_bounds();
                    break;
                }
            }
        }
    }

    // Parse --degree (for Newton/Nova)
    if let Some(pos) = args.iter().position(|a| a == "--degree") {
        if let Some(val) = args.get(pos + 1) {
            if let Ok(d) = val.parse::<u32>() {
                params.poly_degree = d.clamp(2, 8);
            }
        }
    }

    // Parse --supersample (1, 2, or 3)
    if let Some(pos) = args.iter().position(|a| a == "--supersample" || a == "--ss") {
        if let Some(val) = args.get(pos + 1) {
            if let Ok(ss) = val.parse::<u32>() {
                params.supersampling = ss.clamp(1, 3);
            }
        }
    }

    // Parse --palette (classic, oklab, smooth, mono)
    if let Some(pos) = args.iter().position(|a| a == "--palette") {
        if let Some(name) = args.get(pos + 1) {
            params.palette = match name.to_lowercase().as_str() {
                "oklab" => fractals::ColorPalette::Oklab,
                "smooth" => fractals::ColorPalette::Smooth,
                "mono" | "monochrome" => fractals::ColorPalette::Monochrome,
                "thinfilm" | "thin-film" | "film" => fractals::ColorPalette::ThinFilm,
                "aurora" => fractals::ColorPalette::Aurora,
                "storm" => fractals::ColorPalette::Storm,
                "canopy" | "primordial" => fractals::ColorPalette::Canopy,
                "bokeh" | "canopy-bokeh" => fractals::ColorPalette::CanopyBokeh,
                _ => fractals::ColorPalette::Classic,
            };
            params.coloring_param = params.palette.default_param();
        }
    }

    // Parse --no-median / --median (default: median on)
    if args.iter().any(|a| a == "--no-median") {
        params.use_median = false;
    }
    if args.iter().any(|a| a == "--median") {
        params.use_median = true;
    }

    // Parse --iter
    if let Some(pos) = args.iter().position(|a| a == "--iter") {
        if let Some(val) = args.get(pos + 1) {
            if let Ok(i) = val.parse::<u32>() {
                params.max_iter = i.clamp(10, 50000);
            }
        }
    }

    // Parse --bounds x_min,x_max,y_min,y_max
    if let Some(pos) = args.iter().position(|a| a == "--bounds") {
        if let Some(val) = args.get(pos + 1) {
            let parts: Vec<f64> = val.split(',').filter_map(|s| s.parse().ok()).collect();
            if parts.len() == 4 {
                params.set_from_bounds([parts[0], parts[1], parts[2], parts[3]]);
            }
        }
    }

    // Headless wgpu device
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("No GPU adapter found");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("export"),
            ..Default::default()
        },
        None,
    ))
    .expect("Failed to create device");

    println!("Exporting {} to {} ...", params.fractal_type.name(), path);

    // Build a minimal pipeline inline for headless rendering
    // Parse --width and --height (default 1920x1080)
    let width = args.iter().position(|a| a == "--width")
        .and_then(|p| args.get(p + 1))
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1920);
    let height = args.iter().position(|a| a == "--height")
        .and_then(|p| args.get(p + 1))
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1080);
    let ss = params.supersampling;
    // Align width to 64-pixel boundary (wgpu row alignment requirement)
    let align = |w: u32| -> u32 { (w + 63) / 64 * 64 };
    let width = align(width);
    let out_pixels = (width * height) as u64;

    let escape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/escape.wgsl").into()),
    });
    let newton_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/newton.wgsl").into()),
    });
    let colorize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/colorize.wgsl").into()),
    });
    let finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/finalize.wgsl").into()),
    });
    let median_finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/median_finalize.wgsl").into()),
    });

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<GpuParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let use_median = params.use_median;
    let iter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_pixels * 4 * if use_median { gpu::MAX_MEDIAN_SAMPLES as u64 } else { 1 },
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let z_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_pixels * 16, // vec4<f32> per pixel (z.xy, dz_mag, 0)
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let orbit_trap_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_pixels * 16, // vec4<f32> per pixel (min dist to 4 trap points)
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_pixels * 16, // vec4<f32> per pixel
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_pixels * 4, // u32 per output pixel
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let roots_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 8 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_pixels * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Perturbation resources
    let pixel_step = params.pixel_step_x(width);
    let use_perturb = params.fractal_type.is_escape_time()
        && params.fractal_type != FractalType::Multibrot
        && pixel_step < 1e-7;

    let perturb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/escape_perturb.wgsl").into()),
    });
    let ref_orbit_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (params.max_iter as u64 + 1) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let perturb_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<PerturbGpuParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Decompose pixel_step into mantissa + exponent for extended-range perturbation
    let step_y = params.pixel_step_y(height);
    let ps_exp = pixel_step.log2().floor() as i32;
    let ps_scale = 2.0_f64.powi(ps_exp);
    let ps_mantissa_x = (pixel_step / ps_scale) as f32;
    let ps_mantissa_y = (step_y / ps_scale) as f32;

    if use_perturb {
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
        queue.write_buffer(&ref_orbit_buf, 0, bytemuck::cast_slice(&perturb_data.orbit));
        let pgpu = PerturbGpuParams { ref_orbit_len: perturb_data.orbit_len, pixel_step_exp: ps_exp, _pad: [0; 2] };
        queue.write_buffer(&perturb_params_buf, 0, bytemuck::bytes_of(&pgpu));
        println!("  Perturbation mode: ref orbit {} iters, precision for 1e-{:.0}", perturb_data.orbit_len, -pixel_step.log10());
    }

    if params.fractal_type.needs_roots() {
        let roots = params.compute_roots();
        let mut flat: Vec<f32> = roots.iter().flat_map(|r| r.iter().copied()).collect();
        flat.resize(32, 0.0);
        queue.write_buffer(&roots_buf, 0, bytemuck::cast_slice(&flat));
    }

    // Build bind group layouts + pipelines
    let bgl_uniform = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl_storage = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    macro_rules! be {
        ($b:expr, $buf:expr) => {
            wgpu::BindGroupEntry { binding: $b, resource: $buf.as_entire_binding() }
        };
    }

    // Escape pipeline
    let esc_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_uniform(0), bgl_storage(1, false), bgl_storage(2, false), bgl_storage(3, false)],
    });
    let esc_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&esc_layout], push_constant_ranges: &[],
        })),
        module: &escape_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let esc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &esc_layout,
        entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &orbit_trap_buf)],
    });

    // Perturbation pipeline
    let perturb_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_uniform(0), bgl_storage(1, false), bgl_storage(2, false), bgl_storage(3, true), bgl_uniform(4), bgl_storage(5, false)],
    });
    let perturb_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&perturb_layout], push_constant_ranges: &[],
        })),
        module: &perturb_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let perturb_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &perturb_layout,
        entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &ref_orbit_buf), be!(4, &perturb_params_buf), be!(5, &orbit_trap_buf)],
    });

    // Newton pipeline
    let new_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_uniform(0), bgl_storage(1, false), bgl_storage(2, false), bgl_storage(3, true), bgl_storage(4, false)],
    });
    let new_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&new_layout], push_constant_ranges: &[],
        })),
        module: &newton_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let new_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &new_layout,
        entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &roots_buf), be!(4, &orbit_trap_buf)],
    });

    // Colorize pipeline (now writes to accum buffer instead of output)
    let col_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_uniform(0), bgl_storage(1, true), bgl_storage(2, true), bgl_storage(3, false), bgl_storage(4, true), bgl_storage(5, true)],
    });
    let col_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&col_layout], push_constant_ranges: &[],
        })),
        module: &colorize_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let col_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &col_layout,
        entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &accum_buf), be!(4, &roots_buf), be!(5, &orbit_trap_buf)],
    });

    // Finalize pipeline
    let fin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_uniform(0), bgl_storage(1, true), bgl_storage(2, false)],
    });
    let fin_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&fin_layout], push_constant_ranges: &[],
        })),
        module: &finalize_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let fin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &fin_layout,
        entries: &[be!(0, &params_buf), be!(1, &accum_buf), be!(2, &out_buf)],
    });

    // Median finalize pipeline (params, iterations, output, final_z, roots)
    let med_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_uniform(0), bgl_storage(1, true), bgl_storage(2, false), bgl_storage(3, true), bgl_storage(4, true), bgl_storage(5, true)],
    });
    let med_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&med_layout], push_constant_ranges: &[],
        })),
        module: &median_finalize_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let med_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &med_layout,
        entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &out_buf), be!(3, &z_buf), be!(4, &roots_buf), be!(5, &orbit_trap_buf)],
    });

    let wg_x = (width + 15) / 16;
    let wg_y = (height + 15) / 16;

    let mut samples = fractals::compute_samples(ss);
    if use_median && samples.len() > gpu::MAX_MEDIAN_SAMPLES as usize {
        samples.truncate(gpu::MAX_MEDIAN_SAMPLES as usize);
    }
    let num_samples = samples.len() as u32;
    let params_size = std::mem::size_of::<GpuParams>() as u64;

    // Staging buffer for batched sample params
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (samples.len() as u64 + 1) * params_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Stage all sample params
    let base_gpu_params = params.to_gpu_params(width, height, width);
    for (i, &(offset_x, offset_y, weight)) in samples.iter().enumerate() {
        let mut gpu_params = base_gpu_params;
        gpu_params.sample_offset = [offset_x, offset_y];
        gpu_params.sample_weight = weight;
        gpu_params.sample_index = if use_median { i as u32 } else { 0 };
        gpu_params.num_samples = num_samples;
        if use_perturb {
            gpu_params.pixel_step = [ps_mantissa_x, ps_mantissa_y];
        }
        queue.write_buffer(&staging_buf, i as u64 * params_size, bytemuck::bytes_of(&gpu_params));
    }

    // Single command buffer for all work
    let mut encoder = device.create_command_encoder(&Default::default());
    if !use_median {
        encoder.clear_buffer(&accum_buf, 0, None);
    }

    for i in 0..samples.len() {
        encoder.copy_buffer_to_buffer(&staging_buf, i as u64 * params_size, &params_buf, 0, params_size);

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            if use_perturb {
                pass.set_pipeline(&perturb_pipe);
                pass.set_bind_group(0, &perturb_bg, &[]);
            } else if params.fractal_type.is_escape_time() {
                pass.set_pipeline(&esc_pipe);
                pass.set_bind_group(0, &esc_bg, &[]);
            } else {
                pass.set_pipeline(&new_pipe);
                pass.set_bind_group(0, &new_bg, &[]);
            }
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Colorize and accumulate (only in non-median mode)
        if !use_median {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&col_pipe);
            pass.set_bind_group(0, &col_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    }

    // Finalize pass
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
        queue.write_buffer(&staging_buf, fin_offset, bytemuck::bytes_of(&fin_params));
        encoder.copy_buffer_to_buffer(&staging_buf, fin_offset, &params_buf, 0, params_size);
    }
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        if use_median {
            pass.set_pipeline(&med_pipe);
            pass.set_bind_group(0, &med_bg, &[]);
        } else {
            pass.set_pipeline(&fin_pipe);
            pass.set_bind_group(0, &fin_bg, &[]);
        }
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, out_pixels * 4);
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let img = image::RgbaImage::from_raw(width, height, data.to_vec()).unwrap();
    img.save(path).unwrap();
    let ss_info = if ss > 1 { format!(" ({} samples)", samples.len()) } else { String::new() };
    println!("Done: {}x{}{} -> {}", width, height, ss_info, path);

    Ok(())
}
