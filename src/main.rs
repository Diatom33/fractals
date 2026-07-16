mod app;
mod export;
mod fractals;
mod gpu;
mod nebula;
mod shaders;

fn main() -> eframe::Result {
    env_logger::init();

    // Quick CLI export mode: --export <file.png> [--type mandelbrot|julia|etc]
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--zoom-video") {
        if let Some(dir) = args.get(pos + 1) {
            return export_zoom_video(&args, dir);
        }
    }
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
    // Buffers use an aligned stride; padding is stripped before saving.
    let stride = width.div_ceil(64) * 64;

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

    // Same sample region as the GUI paths (nebula.rs / gpu.rs) so CLI and GUI
    // renders of the same view match.
    let sample_min = [-2.5f32, -1.5f32];
    let sample_max = [1.0f32, 1.5f32];

    println!("Nebulabrot export: {}x{} -> {}", width, height, path);
    println!("  Samples: {}, Iters: R={}, G={}, B={}", total_samples, max_iter_r, max_iter_g, max_iter_b);
    println!("  View: [{}, {}] x [{}, {}]", view_min[0], view_max[0], view_min[1], view_max[1]);

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("No GPU adapter found");
    let adapter_limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("nebulabrot export"),
            required_limits: wgpu::Limits {
                max_buffer_size: adapter_limits.max_buffer_size.min(1 << 31),
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                ..Default::default()
            },
            ..Default::default()
        },
        None,
    )).expect("Failed to create device");

    let out_pixels = (stride as u64) * (height as u64);
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
    let num_dispatches = total_samples.div_ceil(samples_per_dispatch) as u32;

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
            resolution: [width, height], stride,
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
        // 99.9th percentile, matching the GUI paths (nebula.rs / gpu.rs) so CLI
        // and GUI renders normalize to the same brightness.
        let exposure = if nonzero.is_empty() { 0 } else {
            nonzero.sort_unstable();
            nonzero[((nonzero.len() as f64 * 0.999) as usize).min(nonzero.len() - 1)]
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
        resolution: [width, height], stride,
        max_r, max_g, max_b, _pad: [0; 2],
    };
    queue.write_buffer(&fin_params_buf, 0, bytemuck::bytes_of(&fin_params));

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&fin_pipe);
        pass.set_bind_group(0, &fin_bg, &[]);
        pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
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
    let pixels = if stride == width {
        data[..(width * height * 4) as usize].to_vec()
    } else {
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * stride * 4) as usize;
            let end = start + (width * 4) as usize;
            out.extend_from_slice(&data[start..end]);
        }
        out
    };
    drop(data);
    readback_buf.unmap();
    let img = image::RgbaImage::from_raw(width, height, pixels).unwrap();
    img.save(path).unwrap();
    println!("Done: {}x{}, {} samples, {:.1}s -> {}", width, height, total_samples, start.elapsed().as_secs_f64(), path);
    Ok(())
}

/// Parse an integer/float flag like `--width 1920`, falling back to a default.
fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|p| args.get(p + 1))
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(default)
}

/// Parse all fractal/view CLI flags shared by --export and --zoom-video:
/// --type, --degree, --power, --supersample/--ss, --palette, --median/
/// --no-median, --iter, --bounds, and the arbitrary-precision deep-zoom trio
/// --center-re/--center-im/--zoom (--zoom-end also accepted as the extent).
fn parse_fractal_args(args: &[String]) -> fractals::FractalParams {
    use fractals::{FractalParams, FractalType};
    use rug::ops::CompleteRound;

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

    // Parse --power (for Multibrot; supports fractional values like 2.5, 3.7)
    if let Some(pos) = args.iter().position(|a| a == "--power") {
        if let Some(val) = args.get(pos + 1) {
            if let Ok(p) = val.parse::<f32>() {
                // Clamp to match UI slider range; negative powers would make
                // u32(round(power)) undefined in the shader's integer path.
                params.power = p.clamp(2.0, 8.0);
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
                "bioluminescence" | "biolum" | "abyss" => fractals::ColorPalette::Bioluminescence,
                "steve" => fractals::ColorPalette::Steve,
                "invertedpair" | "inverted" | "inverted-pair" | "ip" => fractals::ColorPalette::InvertedPair,
                "obsidian" => fractals::ColorPalette::Obsidian,
                "noctilucent" | "noctilucent-clouds" => fractals::ColorPalette::Noctilucent,
                "lichtenberg" | "lightning" => fractals::ColorPalette::Lichtenberg,
                "corona" | "filament" => fractals::ColorPalette::Corona,
                "slotcanyon" | "slot-canyon" | "canyon" => fractals::ColorPalette::SlotCanyon,
                _ => fractals::ColorPalette::Classic,
            };
            params.coloring_param = params.palette.default_param();
            params.coloring_param_2 = params.palette.default_param_2();
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
                params.max_iter = i.clamp(10, 1_000_000);
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

    // Parse --center-re STR --center-im STR --zoom EXPR for arbitrary-precision deep zoom.
    // Uses rug::Float string parsing so we can hit 1e-100+ depths from CLI.
    // --zoom-end is accepted as the extent too (zoom-video's target depth),
    // so precision is sized for the deepest frame.
    let center_re_str = args.iter().position(|a| a == "--center-re").and_then(|p| args.get(p + 1));
    let center_im_str = args.iter().position(|a| a == "--center-im").and_then(|p| args.get(p + 1));
    let zoom_str = args.iter().position(|a| a == "--zoom" || a == "--zoom-end").and_then(|p| args.get(p + 1));
    if let (Some(cre), Some(cim), Some(z)) = (center_re_str, center_im_str, zoom_str) {
        let zoom_extent: f64 = z.parse().unwrap_or(0.0);
        if zoom_extent > 0.0 {
            // Precision needs to comfortably exceed the requested depth.
            let prec_bits = (((-zoom_extent.log10()).max(16.0) * 4.0) as u32 + 64).max(128);
            let parse_or_exit = |arg: &str, flag: &str| match rug::Float::parse(arg) {
                Ok(p) => p.complete(prec_bits),
                Err(e) => {
                    eprintln!("Invalid {flag} value {arg:?}: {e}");
                    std::process::exit(1);
                }
            };
            params.center_re = parse_or_exit(cre.as_str(), "--center-re");
            params.center_im = parse_or_exit(cim.as_str(), "--center-im");
            // Aspect-correct: half_range_x = zoom/2 in x, half_range_y matches view aspect
            let aspect = (args.iter().position(|a| a == "--width").and_then(|p| args.get(p + 1))
                .and_then(|v| v.parse::<u32>().ok()).unwrap_or(1920)) as f64
                / (args.iter().position(|a| a == "--height").and_then(|p| args.get(p + 1))
                .and_then(|v| v.parse::<u32>().ok()).unwrap_or(1080)) as f64;
            params.half_range_x = zoom_extent * 0.5;
            params.half_range_y = zoom_extent * 0.5 / aspect;
        }
    }

    params
}

fn export_cli(args: &[String], path: &str) -> eframe::Result {
    let params = parse_fractal_args(args);
    let width = parse_flag(args, "--width", 1920u32);
    let height = parse_flag(args, "--height", 1080u32);

    let config = export::ExportConfig {
        width,
        height,
        ss: params.supersampling,
        max_iter: None,
        path: path.to_string(),
    };
    match export::export_headless(&params, &config, |msg| println!("{msg}")) {
        Ok(msg) => println!("Done: {msg}"),
        Err(e) => {
            eprintln!("Export failed: {e}");
            std::process::exit(1);
        }
    }
    Ok(())
}

/// Render a zoom video: a sequence of frames zooming into the parsed center,
/// with the zoom extent interpolated geometrically (constant zoom speed) from
/// --zoom-start down to --zoom-end. Frames are written as PNGs into `out_dir`
/// and assembled into zoom.mp4 if ffmpeg is on PATH.
fn export_zoom_video(args: &[String], out_dir: &str) -> eframe::Result {
    let mut params = parse_fractal_args(args);
    let width = parse_flag(args, "--width", 1920u32);
    let height = parse_flag(args, "--height", 1080u32);
    let frames: u32 = parse_flag(args, "--frames", 300u32).max(2);
    let fps: u32 = parse_flag(args, "--fps", 30u32).max(1);
    let zoom_start: f64 = parse_flag(args, "--zoom-start", 4.0f64);
    let zoom_end: f64 = {
        let end = parse_flag(args, "--zoom-end", 0.0f64);
        if end > 0.0 { end } else { parse_flag(args, "--zoom", 0.0f64) }
    };
    if zoom_end <= 0.0 {
        eprintln!("--zoom-video needs a target depth: pass --zoom-end (or --zoom) with the final x-extent, e.g. --zoom-end 1e-12");
        std::process::exit(1);
    }
    if zoom_end >= zoom_start {
        eprintln!("--zoom-end ({zoom_end}) must be smaller than --zoom-start ({zoom_start})");
        std::process::exit(1);
    }

    let dir = export::expand_tilde(out_dir);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("Cannot create output directory {dir}: {e}");
        std::process::exit(1);
    }

    // Precision must cover the deepest frame.
    params.half_range_x = zoom_end * 0.5;
    params.half_range_y = params.half_range_x * (height as f64 / width as f64);
    params.ensure_precision();

    println!(
        "Zoom video: {} frames, {}x{}, zoom {:.3e} -> {:.3e}, {} iters, palette {}",
        frames, width, height, zoom_start, zoom_end, params.max_iter, params.palette.name()
    );

    let mut ctx = match export::ExportContext::new(width, height, params.max_iter, params.use_median) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU setup failed: {e}");
            std::process::exit(1);
        }
    };

    // Geometric zoom schedule: zoom_i = start * ratio^i, constant zoom speed.
    let ratio = (zoom_end / zoom_start).powf(1.0 / (frames as f64 - 1.0));
    let zoom_at = |i: u32| zoom_start * ratio.powi(i as i32);

    // If any frame needs perturbation, build the reference orbit ONCE with the
    // deepest frame's precision and the shallowest perturbed frame's
    // delta_c_max — BLA radii computed for a larger delta_c_max stay valid
    // (conservative) for every deeper frame, so no per-frame recompute.
    let perturbable = params.fractal_type.is_escape_time()
        && params.fractal_type != fractals::FractalType::Multibrot;
    let deepest_step = zoom_end / (width as f64 - 1.0).max(1.0);
    if perturbable && deepest_step < 1e-7 {
        let shallowest_perturbed_zoom = (0..frames)
            .map(zoom_at)
            .find(|z| z / (width as f64 - 1.0).max(1.0) < 1e-7)
            .unwrap_or(zoom_end);
        let half_x = shallowest_perturbed_zoom * 0.5;
        let half_y = half_x * (height as f64 / width as f64);
        let delta_c_max = half_x.hypot(half_y);
        if let Err(e) = ctx.ensure_orbit(&params, deepest_step, delta_c_max, &|msg| println!("  {msg}")) {
            eprintln!("Reference orbit failed: {e}");
            std::process::exit(1);
        }
    }

    let start = std::time::Instant::now();
    let quiet = |_msg: String| {};
    for i in 0..frames {
        let zoom_i = zoom_at(i);
        params.half_range_x = zoom_i * 0.5;
        params.half_range_y = params.half_range_x * (height as f64 / width as f64);

        let pixels = match ctx.render_frame(&params, &quiet) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("\nFrame {i} failed: {e}");
                std::process::exit(1);
            }
        };
        let frame_path = format!("{dir}/frame_{:05}.png", i + 1);
        let img = image::RgbaImage::from_raw(width, height, pixels)
            .expect("pixel buffer size mismatch");
        if let Err(e) = img.save(&frame_path) {
            eprintln!("\nFailed to save {frame_path}: {e}");
            std::process::exit(1);
        }

        let done = i + 1;
        let elapsed = start.elapsed().as_secs_f64();
        let eta = elapsed / done as f64 * (frames - done) as f64;
        print!(
            "\r  Frame {done}/{frames} (zoom {:.3e}, {:.1}s elapsed, ~{:.0}s left)   ",
            zoom_i, elapsed, eta
        );
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!("\nFrames done in {:.1}s -> {dir}/", start.elapsed().as_secs_f64());

    // Assemble with ffmpeg if available; otherwise print the command.
    // yuv420p needs even dimensions, hence the crop filter.
    let mp4 = format!("{dir}/zoom.mp4");
    let ffmpeg_args = [
        "-y",
        "-framerate", &fps.to_string(),
        "-i", &format!("{dir}/frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "16",
        "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
        &mp4,
    ];
    match std::process::Command::new("ffmpeg")
        .args(ffmpeg_args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
    {
        Ok(s) if s.success() => println!("Assembled {mp4}"),
        Ok(s) => eprintln!("ffmpeg exited with {s}; frames are in {dir}/"),
        Err(_) => {
            println!("ffmpeg not found — assemble manually with:");
            println!("  ffmpeg {}", ffmpeg_args.join(" "));
        }
    }
    Ok(())
}
