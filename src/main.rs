mod app;
mod fractals;
mod gpu;

fn main() -> eframe::Result {
    env_logger::init();

    // Quick CLI export mode: --export <file.png> [--type mandelbrot|julia|etc]
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--export") {
        if let Some(path) = args.get(pos + 1) {
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

fn export_cli(args: &[String], path: &str) -> eframe::Result {
    use fractals::{FractalParams, FractalType};

    let mut params = FractalParams::default();

    // Parse --type
    if let Some(pos) = args.iter().position(|a| a == "--type") {
        if let Some(name) = args.get(pos + 1) {
            for &ft in FractalType::ALL {
                if ft.name().to_lowercase().replace(' ', "") == name.to_lowercase().replace(' ', "")
                    || ft.name().to_lowercase().replace(' ', "_") == name.to_lowercase()
                {
                    params.fractal_type = ft;
                    params.bounds = ft.default_bounds();
                    break;
                }
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

    // Build a minimal GpuState-like pipeline inline
    let width = 1920u32;
    let height = 1080u32;
    let gpu_params = params.to_gpu_params(width, height);

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

    let num_pixels = (width * height) as u64;

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of_val(&gpu_params) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let iter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: num_pixels * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let z_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: num_pixels * 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: num_pixels * 4,
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
        size: num_pixels * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&gpu_params));

    if params.fractal_type.needs_roots() {
        let roots = params.compute_roots();
        let mut flat: Vec<f32> = roots.iter().flat_map(|r| r.iter().copied()).collect();
        flat.resize(32, 0.0);
        queue.write_buffer(&roots_buf, 0, bytemuck::cast_slice(&flat));
    }

    // Build bind groups + pipelines
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
        entries: &[bgl_uniform(0), bgl_storage(1, false), bgl_storage(2, false)],
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
        entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf)],
    });

    // Newton pipeline
    let new_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_uniform(0), bgl_storage(1, false), bgl_storage(2, false), bgl_storage(3, true)],
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
        entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &roots_buf)],
    });

    // Colorize pipeline
    let col_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_uniform(0), bgl_storage(1, true), bgl_storage(2, true), bgl_storage(3, false), bgl_storage(4, true)],
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
        entries: &[be!(0, &params_buf), be!(1, &iter_buf), be!(2, &z_buf), be!(3, &out_buf), be!(4, &roots_buf)],
    });

    let wg_x = (width + 15) / 16;
    let wg_y = (height + 15) / 16;

    let mut encoder = device.create_command_encoder(&Default::default());

    // Compute
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        if params.fractal_type.is_escape_time() {
            pass.set_pipeline(&esc_pipe);
            pass.set_bind_group(0, &esc_bg, &[]);
        } else {
            pass.set_pipeline(&new_pipe);
            pass.set_bind_group(0, &new_bg, &[]);
        }
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    // Colorize
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&col_pipe);
        pass.set_bind_group(0, &col_bg, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    // Copy to readback
    encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, num_pixels * 4);

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
    println!("Done: {}x{} → {}", width, height, path);

    Ok(())
}
