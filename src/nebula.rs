/// Nebulabrot export pipeline.
/// Runs on a background thread with its own wgpu device, sampling random c values
/// and accumulating orbit hit-count histograms across R/G/B channels with different
/// iteration limits. Progress is communicated via shared atomics.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Configuration for a Nebulabrot export render.
pub struct NebulaExportConfig {
    pub width: u32,
    pub height: u32,
    pub num_samples: u64,
    pub max_iter_r: u32,
    pub max_iter_g: u32,
    pub max_iter_b: u32,
    pub output_path: String,
    pub batch_size: u32,
}

/// Shared progress state between the export thread and the UI.
pub struct NebulaProgress {
    pub samples_done: AtomicU64,
    pub total_samples: AtomicU64,
    pub cancelled: AtomicBool,
    pub finished: AtomicBool,
    pub error: Mutex<Option<String>>,
    pub result_path: Mutex<Option<String>>,
}

impl NebulaProgress {
    pub fn new(total: u64) -> Self {
        Self {
            samples_done: AtomicU64::new(0),
            total_samples: AtomicU64::new(total),
            cancelled: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            error: Mutex::new(None),
            result_path: Mutex::new(None),
        }
    }
}

/// Align width so bytes_per_row (width * 4) is a multiple of 256.
fn align_width(w: u32) -> u32 {
    (w + 63) / 64 * 64
}

/// GPU uniform matching NebulaParams in nebula_sample.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NebulaGpuParams {
    resolution: [u32; 2],
    stride: u32,
    max_iter_r: u32,
    max_iter_g: u32,
    max_iter_b: u32,
    samples_per_thread: u32,
    dispatch_index: u32,
    sample_min: [f32; 2],
    sample_max: [f32; 2],
    view_min: [f32; 2],
    view_max: [f32; 2],
}

/// GPU uniform matching NebFinParams in nebula_finalize.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NebulaFinParams {
    resolution: [u32; 2],
    stride: u32,
    max_r: u32,
    max_g: u32,
    max_b: u32,
    _pad: [u32; 2],
}

/// Run the full Nebulabrot export on a background thread.
/// Creates its own wgpu device so it doesn't interfere with the UI pipeline.
pub fn run_nebula_export(
    config: NebulaExportConfig,
    progress: Arc<NebulaProgress>,
    ctx: egui::Context,
) {
    std::thread::spawn(move || {
        if let Err(e) = run_nebula_export_inner(&config, &progress, &ctx) {
            *progress.error.lock().unwrap() = Some(e);
        }
        progress.finished.store(true, Ordering::SeqCst);
        ctx.request_repaint();
    });
}

fn run_nebula_export_inner(
    config: &NebulaExportConfig,
    progress: &NebulaProgress,
    ctx: &egui::Context,
) -> Result<(), String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok_or("No GPU adapter found")?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("nebula_export"),
            ..Default::default()
        },
        None,
    ))
    .map_err(|e| format!("Failed to create device: {e}"))?;

    let width = align_width(config.width);
    let height = config.height;
    let num_pixels = (width as u64) * (height as u64);

    let sample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nebula_sample"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nebula_sample.wgsl").into()),
    });
    let finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nebula_finalize"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nebula_finalize.wgsl").into()),
    });

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nebula_params"),
        size: std::mem::size_of::<NebulaGpuParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let hist_size = num_pixels * 4; // u32 per pixel
    let histogram_r = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram_r"),
        size: hist_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let histogram_g = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram_g"),
        size: hist_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let histogram_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram_b"),
        size: hist_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Clear histograms
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&histogram_r, 0, None);
        encoder.clear_buffer(&histogram_g, 0, None);
        encoder.clear_buffer(&histogram_b, 0, None);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Sample pipeline
    let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nebula_sample_layout"),
        entries: &[
            bgl_uniform(0),
            bgl_storage(1, false),
            bgl_storage(2, false),
            bgl_storage(3, false),
        ],
    });
    let sample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nebula_sample"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&sample_bgl],
            push_constant_ranges: &[],
        })),
        module: &sample_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let sample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nebula_sample_bg"),
        layout: &sample_bgl,
        entries: &[
            bg_entry(0, &params_buf),
            bg_entry(1, &histogram_r),
            bg_entry(2, &histogram_g),
            bg_entry(3, &histogram_b),
        ],
    });

    // Sampling loop
    let threads_per_dispatch = config.batch_size;
    let samples_per_thread = 64u32;
    let samples_per_dispatch = threads_per_dispatch as u64 * samples_per_thread as u64;
    let total_samples = config.num_samples;
    let num_dispatches = (total_samples + samples_per_dispatch - 1) / samples_per_dispatch;

    let workgroups = (threads_per_dispatch + 255) / 256;

    // View bounds: full Mandelbrot region for sampling, same for view
    let sample_min = [-2.5_f32, -1.5];
    let sample_max = [1.0_f32, 1.5];
    let view_min = sample_min;
    let view_max = sample_max;

    let repaint_interval = (num_dispatches / 100).max(1);

    for dispatch_idx in 0..num_dispatches {
        if progress.cancelled.load(Ordering::Relaxed) {
            return Err("Cancelled".to_string());
        }

        let gpu_params = NebulaGpuParams {
            resolution: [config.width, height],
            stride: width,
            max_iter_r: config.max_iter_r,
            max_iter_g: config.max_iter_g,
            max_iter_b: config.max_iter_b,
            samples_per_thread,
            dispatch_index: dispatch_idx as u32,
            sample_min,
            sample_max,
            view_min,
            view_max,
        };
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&gpu_params));

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&sample_pipeline);
            pass.set_bind_group(0, &sample_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let done = ((dispatch_idx + 1) * samples_per_dispatch).min(total_samples);
        progress.samples_done.store(done, Ordering::Relaxed);

        if dispatch_idx % repaint_interval == 0 {
            ctx.request_repaint();
        }
    }

    // Read back histograms to find max values for normalization
    let readback_r = create_readback(&device, hist_size);
    let readback_g = create_readback(&device, hist_size);
    let readback_b = create_readback(&device, hist_size);

    {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&histogram_r, 0, &readback_r, 0, hist_size);
        encoder.copy_buffer_to_buffer(&histogram_g, 0, &readback_g, 0, hist_size);
        encoder.copy_buffer_to_buffer(&histogram_b, 0, &readback_b, 0, hist_size);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    let max_r = read_percentile_max(&device, &readback_r, width, config.width, height);
    let max_g = read_percentile_max(&device, &readback_g, width, config.width, height);
    let max_b = read_percentile_max(&device, &readback_b, width, config.width, height);

    // Finalize pipeline: normalize histograms to RGB
    let fin_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nebula_fin_params"),
        size: std::mem::size_of::<NebulaFinParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nebula_output"),
        size: num_pixels * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let fin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nebula_fin_layout"),
        entries: &[
            bgl_uniform(0),
            bgl_storage(1, true),
            bgl_storage(2, true),
            bgl_storage(3, true),
            bgl_storage(4, false),
        ],
    });
    let fin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nebula_finalize"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&fin_bgl],
            push_constant_ranges: &[],
        })),
        module: &finalize_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let fin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nebula_fin_bg"),
        layout: &fin_bgl,
        entries: &[
            bg_entry(0, &fin_params_buf),
            bg_entry(1, &histogram_r),
            bg_entry(2, &histogram_g),
            bg_entry(3, &histogram_b),
            bg_entry(4, &output_buf),
        ],
    });

    let fin_params = NebulaFinParams {
        resolution: [config.width, height],
        stride: width,
        max_r,
        max_g,
        max_b,
        _pad: [0; 2],
    };
    queue.write_buffer(&fin_params_buf, 0, bytemuck::bytes_of(&fin_params));

    let wg_x = (config.width + 15) / 16;
    let wg_y = (height + 15) / 16;
    let out_readback = create_readback(&device, num_pixels * 4);

    {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&fin_pipeline);
            pass.set_bind_group(0, &fin_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &out_readback, 0, num_pixels * 4);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Read back and save PNG
    let slice = out_readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("Map recv error: {e}"))?
        .map_err(|e| format!("Map error: {e}"))?;

    let data = slice.get_mapped_range();

    // Strip padding if stride != display width
    let pixels = if width == config.width {
        data[..(config.width * height * 4) as usize].to_vec()
    } else {
        let mut out = Vec::with_capacity((config.width * height * 4) as usize);
        for row in 0..height {
            let start = (row * width * 4) as usize;
            let end = start + (config.width * 4) as usize;
            out.extend_from_slice(&data[start..end]);
        }
        out
    };
    drop(data);
    out_readback.unmap();

    let img = image::RgbaImage::from_raw(config.width, height, pixels)
        .ok_or("Failed to create image from pixel data")?;

    // Expand ~ in path
    let path = if config.output_path.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            let home = home.to_string_lossy();
            format!("{}{}", home, &config.output_path[1..])
        } else {
            config.output_path.clone()
        }
    } else {
        config.output_path.clone()
    };

    img.save(&path).map_err(|e| format!("Failed to save: {e}"))?;

    *progress.result_path.lock().unwrap() = Some(path);
    Ok(())
}

/// Read histogram data back and compute 99.9th percentile max for exposure.
fn read_percentile_max(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    stride: u32,
    display_width: u32,
    height: u32,
) -> u32 {
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let values: &[u32] = bytemuck::cast_slice(&data);

    // Collect only the visible pixels (skip stride padding)
    let mut visible: Vec<u32> = Vec::with_capacity((display_width * height) as usize);
    for row in 0..height {
        let start = (row * stride) as usize;
        let end = start + display_width as usize;
        for &v in &values[start..end] {
            if v > 0 {
                visible.push(v);
            }
        }
    }
    drop(data);
    readback.unmap();

    if visible.is_empty() {
        return 1;
    }

    visible.sort_unstable();
    let idx = ((visible.len() as f64) * 0.999) as usize;
    let idx = idx.min(visible.len() - 1);
    visible[idx].max(1)
}

fn create_readback(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
