/// Fractal Explorer egui application.

use crate::fractals::{FractalParams, FractalType};
use crate::gpu::GpuState;
use crate::nebula::{NebulaExportConfig, NebulaProgress};
use rug::Float;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Saved view state for undo history and drag operations.
#[derive(Clone)]
struct ViewState {
    center_re: Float,
    center_im: Float,
    half_range_x: f64,
    half_range_y: f64,
}

impl ViewState {
    fn from_params(p: &FractalParams) -> Self {
        ViewState {
            center_re: p.center_re.clone(),
            center_im: p.center_im.clone(),
            half_range_x: p.half_range_x,
            half_range_y: p.half_range_y,
        }
    }

    fn apply_to(&self, p: &mut FractalParams) {
        p.center_re = self.center_re.clone();
        p.center_im = self.center_im.clone();
        p.half_range_x = self.half_range_x;
        p.half_range_y = self.half_range_y;
    }
}

pub struct FractalApp {
    gpu: Option<GpuState>,
    params: FractalParams,
    prev_params_hash: u64,
    history: Vec<ViewState>,
    needs_render: bool,

    // egui-managed texture (CPU readback from GPU)
    texture_handle: Option<egui::TextureHandle>,

    // Drag state: offset texture visually during drag, render on release
    drag_start: Option<egui::Pos2>,
    drag_view: Option<ViewState>,
    drag_pixel_offset: egui::Vec2,

    // Zoom preview: show scaled old texture while waiting for re-render
    // UV rect into the previous texture that corresponds to the new view
    zoom_preview_uv: Option<egui::Rect>,

    // Interactive rendering: SS=1 during rapid input, full SS after idle
    pending_quality_render: bool,
    last_input_time: std::time::Instant,

    // Export
    export_path: String,
    export_msg: String,
    export_msg_is_error: bool,

    // High-res export
    hires_width: u32,
    hires_height: u32,
    hires_ss: u32,
    hires_path: String,
    hires_status: Arc<Mutex<Option<String>>>,
    hires_in_progress: Arc<AtomicBool>,

    // Nebulabrot export
    nebula_width: u32,
    nebula_height: u32,
    nebula_samples_m: f64,
    nebula_iter_r: u32,
    nebula_iter_g: u32,
    nebula_iter_b: u32,
    nebula_batch_size: u32,
    nebula_path: String,
    nebula_progress: Option<Arc<NebulaProgress>>,

    // Coordinate display
    cursor_complex: Option<(f64, f64)>,

    // Last known display dimensions for aspect ratio correction
    last_display_w: u32,
    last_display_h: u32,
}

impl FractalApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Apply custom dark theme
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.window_shadow = egui::epaint::Shadow::NONE;
        style.visuals.panel_fill = egui::Color32::from_rgb(25, 25, 30);
        style.visuals.extreme_bg_color = egui::Color32::from_rgb(15, 15, 20);
        style.visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(35, 35, 42);
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.slider_width = 140.0;
        cc.egui_ctx.set_style(style);

        let gpu = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| GpuState::new(rs));

        Self {
            gpu,
            params: FractalParams::default(),
            prev_params_hash: 0,
            history: Vec::new(),
            needs_render: true,
            texture_handle: None,
            drag_start: None,
            drag_view: None,
            drag_pixel_offset: egui::Vec2::ZERO,
            zoom_preview_uv: None,
            pending_quality_render: false,
            last_input_time: std::time::Instant::now(),
            export_path: String::from("fractal.png"),
            export_msg: String::new(),
            export_msg_is_error: false,
            hires_width: 3840,
            hires_height: 2160,
            hires_ss: 2,
            hires_path: String::from("~/Pictures/fractals/fractal_export.png"),
            hires_status: Arc::new(Mutex::new(None)),
            hires_in_progress: Arc::new(AtomicBool::new(false)),
            nebula_width: 1920,
            nebula_height: 1080,
            nebula_samples_m: 100.0,
            nebula_iter_r: 5000,
            nebula_iter_g: 500,
            nebula_iter_b: 50,
            nebula_batch_size: 65536,
            nebula_path: String::from("~/Pictures/fractals/nebulabrot.png"),
            nebula_progress: None,
            cursor_complex: None,
            last_display_w: 0,
            last_display_h: 0,
        }
    }

    /// Adjust half_range so that the scale (units per pixel) is equal in x and y,
    /// keeping the center of the view fixed. The larger scale wins so nothing
    /// gets cropped — the smaller dimension is expanded to match.
    fn correct_aspect_ratio(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let scale_x = (2.0 * self.params.half_range_x) / width as f64;
        let scale_y = (2.0 * self.params.half_range_y) / height as f64;
        let scale = scale_x.max(scale_y);

        self.params.half_range_x = scale * width as f64 / 2.0;
        self.params.half_range_y = scale * height as f64 / 2.0;

        self.last_display_w = width;
        self.last_display_h = height;
    }

    /// Scale half_range proportionally when the display dimensions change.
    fn scale_bounds_to_new_size(&mut self, new_w: u32, new_h: u32) {
        if self.last_display_w == 0 || self.last_display_h == 0 || new_w == 0 || new_h == 0 {
            return;
        }
        self.params.half_range_x *= new_w as f64 / self.last_display_w as f64;
        self.params.half_range_y *= new_h as f64 / self.last_display_h as f64;

        self.last_display_w = new_w;
        self.last_display_h = new_h;
    }

    fn push_history(&mut self) {
        self.history.push(ViewState::from_params(&self.params));
    }

    /// Mark that we're in an interactive operation — render at SS=1 now,
    /// schedule full-quality re-render after input settles.
    fn mark_interactive(&mut self) {
        if self.params.supersampling > 1 {
            self.pending_quality_render = true;
            self.last_input_time = std::time::Instant::now();
        }
    }

    fn params_hash(&self) -> u64 {
        let (dw, h, stride) = self.gpu.as_ref().map_or((960, 720, 960), |g| {
            (g.display_width, g.height, g.width)
        });
        let binding = self.params.to_gpu_params(dw, h, stride);
        let bytes = bytemuck::bytes_of(&binding);
        let mut hash = 0u64;
        for chunk in bytes.chunks(8) {
            let mut arr = [0u8; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            hash ^= u64::from_le_bytes(arr);
        }
        hash ^= self.params.use_median as u64;
        hash ^= (self.params.coloring_param.to_bits() as u64) << 32;
        if self.params.fractal_type.is_nebulabrot() {
            hash ^= self.params.nebula_iter_r as u64;
            hash ^= (self.params.nebula_iter_g as u64) << 16;
            hash ^= (self.params.nebula_iter_b as u64) << 32;
            hash ^= self.params.nebula_samples_m.to_bits();
        }
        hash
    }
}

/// Section header with consistent styling.
fn section_header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(text)
            .strong()
            .size(13.0)
            .color(egui::Color32::from_rgb(160, 180, 220)),
    );
    ui.separator();
}

impl eframe::App for FractalApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── Ctrl+Q to quit ───────────────────────────────────────────────
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Q)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        // ── Side panel: controls ──────────────────────────────────────────
        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(220.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Fractal Explorer")
                            .heading()
                            .color(egui::Color32::from_rgb(200, 210, 240)),
                    );
                });
                ui.add_space(2.0);

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // ── Fractal Type ──────────────────────────────
                        section_header(ui, "Fractal Type");
                        let prev_type = self.params.fractal_type;
                        egui::ComboBox::from_id_salt("fractal_type")
                            .width(ui.available_width() - 8.0)
                            .selected_text(self.params.fractal_type.name())
                            .show_ui(ui, |ui| {
                                for &ft in FractalType::ALL {
                                    ui.selectable_value(
                                        &mut self.params.fractal_type,
                                        ft,
                                        ft.name(),
                                    );
                                }
                            });
                        if self.params.fractal_type != prev_type {
                            self.params.set_from_default_bounds();
                            self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
                            self.history.clear();
                            self.needs_render = true;
                        }

                        // ── Parameters ────────────────────────────────
                        section_header(ui, "Parameters");

                        ui.label("Max iterations:");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.params.max_iter, 10..=50000)
                                    .logarithmic(true),
                            )
                            .changed()
                        {
                            self.needs_render = true;
                        }

                        ui.add_space(2.0);
                        ui.label("Supersampling:");
                        let ss_label = match self.params.supersampling {
                            2 => "4x4 (16 samples)",
                            3 => "8x8 (64 samples)",
                            _ => "Off",
                        };
                        let prev_ss = self.params.supersampling;
                        egui::ComboBox::from_id_salt("supersampling")
                            .width(ui.available_width() - 8.0)
                            .selected_text(ss_label)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.params.supersampling,
                                    1,
                                    "Off",
                                );
                                ui.selectable_value(
                                    &mut self.params.supersampling,
                                    2,
                                    "4x4 (16 samples)",
                                );
                                ui.selectable_value(
                                    &mut self.params.supersampling,
                                    3,
                                    "8x8 (64 samples)",
                                );
                            });
                        if self.params.supersampling != prev_ss {
                            self.needs_render = true;
                        }

                        // AA filter mode
                        let prev_median = self.params.use_median;
                        ui.add_space(2.0);
                        ui.label("AA filter:");
                        egui::ComboBox::from_id_salt("aa_filter")
                            .width(ui.available_width() - 8.0)
                            .selected_text(if self.params.use_median { "Median" } else { "Accumulate" })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.params.use_median, false, "Accumulate");
                                ui.selectable_value(&mut self.params.use_median, true, "Median");
                            });
                        if self.params.use_median != prev_median {
                            self.needs_render = true;
                        }

                        ui.add_space(2.0);
                        ui.label("Color palette:");
                        let prev_palette = self.params.palette;
                        egui::ComboBox::from_id_salt("palette")
                            .width(ui.available_width() - 8.0)
                            .selected_text(self.params.palette.name())
                            .show_ui(ui, |ui| {
                                for &p in crate::fractals::ColorPalette::ALL {
                                    ui.selectable_value(
                                        &mut self.params.palette,
                                        p,
                                        p.name(),
                                    );
                                }
                            });
                        if self.params.palette != prev_palette {
                            self.params.coloring_param = self.params.palette.default_param();
                            self.needs_render = true;
                        }

                        // Palette-specific parameter slider
                        if self.params.palette.has_param() {
                            ui.add_space(2.0);
                            ui.label(self.params.palette.param_label());
                            let range = match self.params.palette {
                                crate::fractals::ColorPalette::ThinFilm => 0.5..=12.0,
                                crate::fractals::ColorPalette::Aurora => 0.5..=10.0,
                                crate::fractals::ColorPalette::Storm => 0.2..=5.0,
                                crate::fractals::ColorPalette::Bioluminescence => 0.5..=20.0,
                                _ => 0.0..=1.0,
                            };
                            if ui.add(
                                egui::Slider::new(&mut self.params.coloring_param, range)
                                    .step_by(0.1)
                            ).changed() {
                                self.needs_render = true;
                            }
                        }

                        let controls = self.params.fractal_type.visible_controls();

                        if controls.power {
                            ui.add_space(2.0);
                            ui.label("Power d:");
                            if ui
                                .add(
                                    egui::Slider::new(&mut self.params.power, 2.0..=8.0)
                                        .step_by(0.1),
                                )
                                .changed()
                            {
                                self.needs_render = true;
                            }
                        }

                        if controls.julia_c {
                            ui.add_space(2.0);
                            ui.label("Julia c (real):");
                            if ui
                                .add(
                                    egui::Slider::new(&mut self.params.julia_c[0], -2.0..=2.0)
                                        .step_by(0.001)
                                        .max_decimals(4),
                                )
                                .changed()
                            {
                                self.needs_render = true;
                            }
                            ui.label("Julia c (imag):");
                            if ui
                                .add(
                                    egui::Slider::new(&mut self.params.julia_c[1], -2.0..=2.0)
                                        .step_by(0.001)
                                        .max_decimals(4),
                                )
                                .changed()
                            {
                                self.needs_render = true;
                            }
                        }

                        if controls.poly_degree {
                            ui.add_space(2.0);
                            ui.label("Degree n (z^n - 1):");
                            let mut deg = self.params.poly_degree as i32;
                            if ui.add(egui::Slider::new(&mut deg, 2..=8)).changed() {
                                self.params.poly_degree = deg as u32;
                                self.needs_render = true;
                            }
                        }

                        if controls.relaxation {
                            ui.add_space(2.0);
                            ui.label("Relaxation a:");
                            if ui
                                .add(
                                    egui::Slider::new(&mut self.params.relaxation, 0.1..=2.0)
                                        .step_by(0.05),
                                )
                                .changed()
                            {
                                self.needs_render = true;
                            }
                        }

                        if self.params.fractal_type.is_nebulabrot() {
                            ui.add_space(4.0);
                            section_header(ui, "Nebulabrot");
                            ui.horizontal(|ui| {
                                ui.label("R iters:");
                                if ui.add(egui::DragValue::new(&mut self.params.nebula_iter_r).range(10..=50000).speed(10)).changed() {
                                    self.needs_render = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("G iters:");
                                if ui.add(egui::DragValue::new(&mut self.params.nebula_iter_g).range(10..=50000).speed(10)).changed() {
                                    self.needs_render = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("B iters:");
                                if ui.add(egui::DragValue::new(&mut self.params.nebula_iter_b).range(10..=50000).speed(10)).changed() {
                                    self.needs_render = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Samples (M):");
                                if ui.add(egui::DragValue::new(&mut self.params.nebula_samples_m).range(1.0..=1000.0).speed(1.0).max_decimals(0)).changed() {
                                    self.needs_render = true;
                                }
                            });
                        }

                        // ── Navigation ────────────────────────────────
                        section_header(ui, "Navigation");

                        ui.horizontal(|ui| {
                            if ui.button("Reset View").clicked() {
                                self.params.set_from_default_bounds();
                                self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
                                self.history.clear();
                                self.needs_render = true;
                            }
                            let undo_enabled = !self.history.is_empty();
                            if ui
                                .add_enabled(undo_enabled, egui::Button::new("Undo"))
                                .clicked()
                            {
                                if let Some(prev) = self.history.pop() {
                                    prev.apply_to(&mut self.params);
                                    self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
                                    self.needs_render = true;
                                }
                            }
                        });

                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new("Scroll: zoom | Drag: pan | R: reset")
                                .small()
                                .weak(),
                        );

                        // Show current view bounds (truncated for compactness)
                        let [x_min, x_max, y_min, y_max] = self.params.bounds_f64();
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "x: [{:.4}, {:.4}]\ny: [{:.4}, {:.4}]",
                                x_min, x_max, y_min, y_max
                            ))
                            .small()
                            .weak()
                            .monospace(),
                        );

                        if let Some((cx, cy)) = self.cursor_complex {
                            ui.label(
                                egui::RichText::new(format!("cursor: {:.6} + {:.6}i", cx, cy))
                                    .small()
                                    .weak()
                                    .monospace(),
                            );
                        }

                        // Precise center + zoom for sharing / CLI use.
                        // Number of decimals shown scales with zoom depth so the
                        // string preserves enough precision to round-trip the view.
                        let zoom = 2.0 * self.params.half_range_x;
                        let zoom_log10 = if zoom > 0.0 {
                            (-zoom.log10()).max(0.0)
                        } else { 0.0 };
                        let precision_digits = (zoom_log10 as usize + 8).max(20);
                        let center_re_str = self.params.center_re
                            .to_string_radix(10, Some(precision_digits));
                        let center_im_str = self.params.center_im
                            .to_string_radix(10, Some(precision_digits));
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Precise coordinates").small().weak());
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("re:").small().weak().monospace());
                            ui.add(egui::TextEdit::singleline(&mut center_re_str.clone())
                                .desired_width(ui.available_width() - 50.0)
                                .font(egui::TextStyle::Monospace));
                            if ui.small_button("Copy").clicked() {
                                ui.ctx().copy_text(center_re_str.clone());
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("im:").small().weak().monospace());
                            ui.add(egui::TextEdit::singleline(&mut center_im_str.clone())
                                .desired_width(ui.available_width() - 50.0)
                                .font(egui::TextStyle::Monospace));
                            if ui.small_button("Copy").clicked() {
                                ui.ctx().copy_text(center_im_str.clone());
                            }
                        });
                        ui.horizontal(|ui| {
                            let zoom_str = format!("{:.6e}", zoom);
                            ui.label(egui::RichText::new("zoom:").small().weak().monospace());
                            ui.label(egui::RichText::new(&zoom_str).small().monospace());
                            if ui.small_button("Copy CLI args").clicked() {
                                let cli = format!(
                                    "--center-re \"{}\" --center-im \"{}\" --zoom {}",
                                    center_re_str, center_im_str, zoom_str
                                );
                                ui.ctx().copy_text(cli);
                            }
                        });

                        // ── Export ─────────────────────────────────────
                        section_header(ui, "Export");

                        ui.horizontal(|ui| {
                            let te_response = ui.add(
                                egui::TextEdit::singleline(&mut self.export_path)
                                    .desired_width(ui.available_width() - 60.0),
                            );
                            if te_response.changed() {
                                self.export_msg.clear();
                            }
                            if ui.button("Save").clicked() {
                                if let Some(gpu) = &self.gpu {
                                    match gpu.export_png(&self.export_path) {
                                        Ok(()) => {
                                            self.export_msg =
                                                format!("Saved to {}", self.export_path);
                                            self.export_msg_is_error = false;
                                        }
                                        Err(e) => {
                                            self.export_msg = format!("Error: {e}");
                                            self.export_msg_is_error = true;
                                        }
                                    }
                                }
                            }
                        });

                        if !self.export_msg.is_empty() {
                            let color = if self.export_msg_is_error {
                                egui::Color32::from_rgb(255, 120, 120)
                            } else {
                                egui::Color32::from_rgb(120, 255, 120)
                            };
                            ui.label(
                                egui::RichText::new(&self.export_msg).small().color(color),
                            );
                        }

                        // ── High-Res Export ────────────────────────────
                        section_header(ui, "High-Res Export");

                        ui.horizontal(|ui| {
                            ui.label("Width:");
                            ui.add(egui::DragValue::new(&mut self.hires_width).range(64..=15360).speed(16));
                            ui.label("Height:");
                            ui.add(egui::DragValue::new(&mut self.hires_height).range(64..=8640).speed(16));
                        });

                        ui.add_space(2.0);
                        let hires_ss_label = match self.hires_ss {
                            2 => "4x4 (16 samples)",
                            3 => "8x8 (64 samples)",
                            _ => "Off",
                        };
                        ui.horizontal(|ui| {
                            ui.label("SS:");
                            egui::ComboBox::from_id_salt("hires_ss")
                                .width(ui.available_width() - 8.0)
                                .selected_text(hires_ss_label)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.hires_ss, 1, "Off");
                                    ui.selectable_value(&mut self.hires_ss, 2, "4x4 (16 samples)");
                                    ui.selectable_value(&mut self.hires_ss, 3, "8x8 (64 samples)");
                                });
                        });

                        ui.add_space(2.0);
                        ui.add(
                            egui::TextEdit::singleline(&mut self.hires_path)
                                .desired_width(ui.available_width() - 8.0),
                        );

                        ui.add_space(4.0);
                        let in_progress = self.hires_in_progress.load(Ordering::Relaxed);
                        let export_btn = ui.add_enabled(
                            !in_progress,
                            egui::Button::new(if in_progress { "Exporting..." } else { "Export Hi-Res" })
                                .min_size(egui::vec2(ui.available_width() - 8.0, 24.0)),
                        );
                        if export_btn.clicked() {
                            let params = self.params.clone();
                            let config = crate::export::ExportConfig {
                                width: self.hires_width,
                                height: self.hires_height,
                                ss: self.hires_ss,
                                max_iter: None,
                                path: self.hires_path.clone(),
                            };
                            let status = self.hires_status.clone();
                            let in_prog = self.hires_in_progress.clone();
                            let ctx_clone = ctx.clone();
                            in_prog.store(true, Ordering::Relaxed);
                            *status.lock().unwrap() = Some("Starting export...".to_string());
                            std::thread::spawn(move || {
                                let status_cb = {
                                    let status = status.clone();
                                    let ctx = ctx_clone.clone();
                                    move |msg: String| {
                                        *status.lock().unwrap() = Some(msg);
                                        ctx.request_repaint();
                                    }
                                };
                                match crate::export::export_headless(&params, &config, status_cb) {
                                    Ok(msg) => *status.lock().unwrap() = Some(format!("Done: {msg}")),
                                    Err(e) => *status.lock().unwrap() = Some(format!("ERROR: {e}")),
                                }
                                in_prog.store(false, Ordering::Relaxed);
                                ctx_clone.request_repaint();
                            });
                        }

                        if in_progress {
                            ctx.request_repaint_after(std::time::Duration::from_millis(200));
                        }

                        if let Some(status_msg) = self.hires_status.lock().unwrap().as_ref() {
                            let color = if status_msg.starts_with("ERROR") {
                                egui::Color32::from_rgb(255, 120, 120)
                            } else if status_msg.starts_with("Done") {
                                egui::Color32::from_rgb(120, 255, 120)
                            } else {
                                egui::Color32::from_rgb(180, 180, 220)
                            };
                            ui.label(egui::RichText::new(status_msg).small().color(color));
                        }

                        // ── Nebulabrot Export ────────────────────────────
                        section_header(ui, "Nebulabrot Export");

                        ui.horizontal(|ui| {
                            ui.label("Width:");
                            ui.add(egui::DragValue::new(&mut self.nebula_width).range(64..=15360).speed(16));
                            ui.label("Height:");
                            ui.add(egui::DragValue::new(&mut self.nebula_height).range(64..=8640).speed(16));
                        });

                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            ui.label("Samples (M):");
                            ui.add(egui::DragValue::new(&mut self.nebula_samples_m).range(1.0..=10000.0).speed(1.0).max_decimals(0));
                        });

                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            ui.label("R iters:");
                            ui.add(egui::DragValue::new(&mut self.nebula_iter_r).range(10..=50000).speed(10));
                        });
                        ui.horizontal(|ui| {
                            ui.label("G iters:");
                            ui.add(egui::DragValue::new(&mut self.nebula_iter_g).range(10..=50000).speed(10));
                        });
                        ui.horizontal(|ui| {
                            ui.label("B iters:");
                            ui.add(egui::DragValue::new(&mut self.nebula_iter_b).range(10..=50000).speed(10));
                        });

                        ui.add_space(2.0);
                        ui.add(
                            egui::TextEdit::singleline(&mut self.nebula_path)
                                .desired_width(ui.available_width() - 8.0),
                        );

                        ui.add_space(4.0);
                        let nebula_running = self.nebula_progress.as_ref()
                            .map(|p| !p.finished.load(Ordering::Relaxed))
                            .unwrap_or(false);

                        ui.horizontal(|ui| {
                            let btn_label = if nebula_running { "Rendering..." } else { "Render Nebulabrot" };
                            let render_btn = ui.add_enabled(
                                !nebula_running,
                                egui::Button::new(btn_label),
                            );
                            if render_btn.clicked() {
                                // Compute aspect-correct view bounds from current view
                                let cx = self.params.center_re.to_f64();
                                let cy = self.params.center_im.to_f64();
                                let aspect = self.nebula_width as f64 / self.nebula_height as f64;
                                let half_y = self.params.half_range_y;
                                let half_x = half_y * aspect;

                                // Check f32 precision: view range must be representable
                                let view_range = (2.0 * half_x).min(2.0 * half_y);
                                if view_range < 1e-5 {
                                    let progress = Arc::new(NebulaProgress::new(1));
                                    *progress.error.lock().unwrap() = Some(
                                        "View too deep for Nebulabrot (f32 precision limit). Zoom out.".to_string()
                                    );
                                    progress.finished.store(true, Ordering::SeqCst);
                                    self.nebula_progress = Some(progress);
                                } else {
                                    let view_min = [(cx - half_x) as f32, (cy - half_y) as f32];
                                    let view_max = [(cx + half_x) as f32, (cy + half_y) as f32];

                                    let num_samples = (self.nebula_samples_m * 1_000_000.0) as u64;
                                    let config = NebulaExportConfig {
                                        width: self.nebula_width,
                                        height: self.nebula_height,
                                        num_samples,
                                        max_iter_r: self.nebula_iter_r,
                                        max_iter_g: self.nebula_iter_g,
                                        max_iter_b: self.nebula_iter_b,
                                        output_path: self.nebula_path.clone(),
                                        batch_size: self.nebula_batch_size,
                                        view_min,
                                        view_max,
                                    };
                                    let progress = Arc::new(NebulaProgress::new(num_samples));
                                    self.nebula_progress = Some(progress.clone());
                                    crate::nebula::run_nebula_export(config, progress, ctx.clone());
                                }
                            }

                            if nebula_running {
                                if ui.button("Cancel").clicked() {
                                    if let Some(p) = &self.nebula_progress {
                                        p.cancelled.store(true, Ordering::Relaxed);
                                    }
                                }
                            }
                        });

                        if let Some(progress) = &self.nebula_progress {
                            let done = progress.samples_done.load(Ordering::Relaxed);
                            let total = progress.total_samples.load(Ordering::Relaxed).max(1);
                            let frac = done as f32 / total as f32;

                            if nebula_running {
                                ui.add(egui::ProgressBar::new(frac).text(format!("{:.1}%", frac * 100.0)));
                                ctx.request_repaint_after(std::time::Duration::from_millis(200));
                            }

                            if progress.finished.load(Ordering::Relaxed) {
                                if let Some(err) = progress.error.lock().unwrap().as_ref() {
                                    if err == "Cancelled" {
                                        ui.label(egui::RichText::new("Cancelled").small().color(egui::Color32::from_rgb(255, 200, 100)));
                                    } else {
                                        ui.label(egui::RichText::new(format!("ERROR: {err}")).small().color(egui::Color32::from_rgb(255, 120, 120)));
                                    }
                                } else if let Some(path) = progress.result_path.lock().unwrap().as_ref() {
                                    ui.label(egui::RichText::new(format!("Saved: {path}")).small().color(egui::Color32::from_rgb(120, 255, 120)));
                                }
                            }
                        }
                    });
            });

        // ── Bottom panel: status bar ──────────────────────────────────────
        egui::TopBottomPanel::bottom("status")
            .exact_height(24.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    if let Some(gpu) = &self.gpu {
                        ui.spacing_mut().item_spacing.x = 16.0;
                        ui.monospace(
                            egui::RichText::new(self.params.fractal_type.name())
                                .color(egui::Color32::from_rgb(130, 170, 255)),
                        );
                        ui.separator();
                        ui.monospace(format!("{}x{}", gpu.display_width, gpu.height));
                        if self.params.supersampling > 1 {
                            let ss_label = if self.pending_quality_render {
                                format!("{}x SS*", self.params.supersampling)
                            } else {
                                format!("{}x SS", self.params.supersampling)
                            };
                            ui.monospace(ss_label);
                        }
                        ui.separator();
                        ui.monospace(format!("{:.1} ms", gpu.last_render_ms));
                        ui.separator();
                        ui.monospace(egui::RichText::new(&gpu.gpu_name).weak());
                        ui.separator();
                        ui.monospace(
                            egui::RichText::new(format!("{} iters", self.params.max_iter)).weak(),
                        );
                        ui.separator();
                        let pixel_step = self.params.pixel_step_x(gpu.display_width);
                        let zoom_exp = -(pixel_step.log10().floor() as i32);
                        ui.monospace(
                            egui::RichText::new(format!("1e-{}", zoom_exp)).weak(),
                        );
                        if self.params.use_median && self.params.supersampling > 1 {
                            ui.monospace(
                                egui::RichText::new("MEDIAN")
                                    .color(egui::Color32::from_rgb(150, 220, 255)),
                            );
                        }
                        if gpu.using_perturbation {
                            ui.monospace(
                                egui::RichText::new("PERTURB")
                                    .color(egui::Color32::from_rgb(255, 200, 100)),
                            );
                        }
                    } else {
                        ui.monospace("GPU not initialized");
                    }
                });
            });

        // ── Central panel: fractal image ──────────────────────────────────
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
            .show(ctx, |ui| {
                let available = ui.available_size();
                let ppp = ctx.pixels_per_point();
                let display_w = ((available.x * ppp) as u32).max(64);
                let display_h = ((available.y * ppp) as u32).max(64);

                // Resize GPU buffers/texture if needed
                let mut new_display_dims: Option<(u32, u32)> = None;
                if let Some(gpu) = &mut self.gpu {
                    let size_changed = gpu.display_width != display_w || gpu.height != display_h;

                    if size_changed {
                        gpu.resize(display_w, display_h);
                        self.needs_render = true;
                        new_display_dims = Some((gpu.display_width, gpu.height));
                    }
                }
                if let Some((new_w, new_h)) = new_display_dims {
                    if self.last_display_w > 0 {
                        self.scale_bounds_to_new_size(new_w, new_h);
                    } else {
                        self.correct_aspect_ratio(new_w, new_h);
                    }
                }

                // Schedule deferred quality render after interaction settles
                if self.pending_quality_render
                    && self.last_input_time.elapsed()
                        > std::time::Duration::from_millis(150)
                {
                    self.pending_quality_render = false;
                    self.needs_render = true;
                }
                if self.pending_quality_render {
                    ctx.request_repaint_after(std::time::Duration::from_millis(150));
                }

                // Render if needed (params changed or flagged)
                let hash = if self.needs_render { 0 } else { self.params_hash() };
                let did_render = self.needs_render || hash != self.prev_params_hash;
                if did_render {
                    if let Some(gpu) = &mut self.gpu {
                        if self.params.fractal_type.is_nebulabrot() {
                            let cx = self.params.center_re.to_f64();
                            let cy = self.params.center_im.to_f64();
                            let aspect = gpu.display_width as f64 / gpu.height as f64;
                            let half_y = self.params.half_range_y;
                            let half_x = half_y * aspect;
                            let view_min = [(cx - half_x) as f32, (cy - half_y) as f32];
                            let view_max = [(cx + half_x) as f32, (cy + half_y) as f32];
                            let dispatches = if self.pending_quality_render {
                                ((self.params.nebula_samples_m * 1_000_000.0 / (65536.0 * 64.0)) as u32).max(1) / 4
                            } else {
                                ((self.params.nebula_samples_m * 1_000_000.0 / (65536.0 * 64.0)) as u32).max(1)
                            }.max(1);
                            gpu.render_nebulabrot(
                                view_min, view_max,
                                self.params.nebula_iter_r, self.params.nebula_iter_g, self.params.nebula_iter_b,
                                dispatches,
                            );
                        } else {
                            // Use SS=1 during interactive operations for responsiveness
                            let render_ss = if self.pending_quality_render {
                                1
                            } else {
                                self.params.supersampling
                            };
                            gpu.render(&self.params, render_ss, self.params.use_median);
                        }
                    }
                    self.prev_params_hash = self.params_hash();
                    self.needs_render = false;
                }

                // CPU readback → egui::ColorImage texture upload
                if did_render {
                    if let Some(gpu) = &self.gpu {
                        let pixels = gpu.read_pixels();
                        let image = egui::ColorImage::from_rgba_unmultiplied(
                            [gpu.display_width as usize, gpu.height as usize],
                            &pixels,
                        );
                        match &mut self.texture_handle {
                            Some(h) => h.set(image, egui::TextureOptions::LINEAR),
                            None => {
                                self.texture_handle = Some(ctx.load_texture(
                                    "fractal",
                                    image,
                                    egui::TextureOptions::LINEAR,
                                ));
                            }
                        }
                    }
                    // Fresh render matches the current view — clear zoom preview
                    self.zoom_preview_uv = None;
                }

                // Display image filling the panel
                if let Some(handle) = &self.texture_handle {
                    let size = egui::vec2(available.x, available.y);
                    let (response, _painter) =
                        ui.allocate_painter(size, egui::Sense::click_and_drag());
                    let rect = response.rect;

                    // Offset texture during drag for instant visual feedback
                    let draw_rect = rect.translate(self.drag_pixel_offset);
                    let uv_rect = self.zoom_preview_uv.unwrap_or(
                        egui::Rect::from_min_max(
                            egui::pos2(0.0, 0.0),
                            egui::pos2(1.0, 1.0),
                        ),
                    );
                    ui.painter().image(
                        handle.id(),
                        draw_rect,
                        uv_rect,
                        egui::Color32::WHITE,
                    );

                    // Update cursor complex coordinates
                    if let Some(pos) = response.hover_pos() {
                        let frac_x =
                            ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0) as f64;
                        let frac_y =
                            ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0) as f64;
                        let cx = self.params.center_re.to_f64()
                            + (frac_x - 0.5) * 2.0 * self.params.half_range_x;
                        let cy = self.params.center_im.to_f64()
                            + (frac_y - 0.5) * 2.0 * self.params.half_range_y;
                        self.cursor_complex = Some((cx, cy));
                    } else {
                        self.cursor_complex = None;
                    }

                    // Mouse & keyboard interaction
                    self.handle_input(ctx, &response, rect);
                }
            });
    }
}

impl FractalApp {
    fn handle_input(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        rect: egui::Rect,
    ) {
        // ── Scroll zoom toward cursor ─────────────────────────────────
        if response.hovered() {
            let scroll = ctx.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                if let Some(pos) = response.hover_pos() {
                    let frac_x = ((pos.x - rect.min.x) / rect.width()) as f64;
                    let frac_y = ((pos.y - rect.min.y) / rect.height()) as f64;

                    let factor: f64 = if scroll > 0.0 { 0.85 } else { 1.0 / 0.85 };

                    self.push_history();
                    self.params.ensure_precision();

                    let offset_re = (frac_x - 0.5) * 2.0 * self.params.half_range_x;
                    let offset_im = (frac_y - 0.5) * 2.0 * self.params.half_range_y;

                    let shift_re = Float::with_val(
                        self.params.center_re.prec(),
                        offset_re * (1.0 - factor),
                    );
                    let shift_im = Float::with_val(
                        self.params.center_im.prec(),
                        offset_im * (1.0 - factor),
                    );
                    self.params.center_re += shift_re;
                    self.params.center_im += shift_im;
                    self.params.half_range_x *= factor;
                    self.params.half_range_y *= factor;

                    // Compute zoom preview UV rect for instant visual feedback.
                    // In the new view, normalized position p maps to old-view
                    // position: nx + (p - nx) * factor (and similarly for y).
                    // So UV min (p=0) = nx*(1-factor), UV max (p=1) = nx + (1-nx)*factor.
                    let nx = frac_x as f32;
                    let ny = frac_y as f32;
                    let f = factor as f32;
                    let new_uv = egui::Rect::from_min_max(
                        egui::pos2(nx * (1.0 - f), ny * (1.0 - f)),
                        egui::pos2(nx + (1.0 - nx) * f, ny + (1.0 - ny) * f),
                    );
                    // Compose with existing preview UV if user scrolls multiple
                    // times before a render completes.
                    self.zoom_preview_uv = Some(match self.zoom_preview_uv {
                        Some(prev) => {
                            // Map new_uv through prev: result.min = prev.min + new_uv.min * prev.size()
                            let pw = prev.width();
                            let ph = prev.height();
                            egui::Rect::from_min_max(
                                egui::pos2(
                                    prev.min.x + new_uv.min.x * pw,
                                    prev.min.y + new_uv.min.y * ph,
                                ),
                                egui::pos2(
                                    prev.min.x + new_uv.max.x * pw,
                                    prev.min.y + new_uv.max.y * ph,
                                ),
                            )
                        }
                        None => new_uv,
                    });

                    self.mark_interactive();
                    self.needs_render = true;
                }
            }
        }

        // ── Drag to pan (deferred: offset texture visually, render on release) ──
        if response.drag_started() {
            self.drag_start = response.interact_pointer_pos();
            self.drag_view = Some(ViewState::from_params(&self.params));
        }
        if response.dragged() {
            if let Some(start) = self.drag_start {
                if let Some(current) = response.interact_pointer_pos() {
                    self.drag_pixel_offset =
                        egui::vec2(current.x - start.x, current.y - start.y);
                    // No render — just visual offset of existing texture
                }
            }
        }
        if response.drag_stopped() {
            if let Some(orig) = self.drag_view.take() {
                let dx_px = self.drag_pixel_offset.x as f64;
                let dy_px = self.drag_pixel_offset.y as f64;
                if dx_px != 0.0 || dy_px != 0.0 {
                    let dx = -dx_px / rect.width() as f64 * (2.0 * orig.half_range_x);
                    let dy = -dy_px / rect.height() as f64 * (2.0 * orig.half_range_y);
                    self.params.ensure_precision();
                    self.params.center_re = Float::with_val(
                        orig.center_re.prec(),
                        &orig.center_re + dx,
                    );
                    self.params.center_im = Float::with_val(
                        orig.center_im.prec(),
                        &orig.center_im + dy,
                    );
                    self.history.push(orig);
                    self.mark_interactive();
                    self.needs_render = true;
                }
            }
            self.drag_start = None;
            self.drag_pixel_offset = egui::Vec2::ZERO;
        }

        // ── Double-click to reset ─────────────────────────────────────
        if response.double_clicked() {
            self.params.set_from_default_bounds();
            self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
            self.history.clear();
            self.needs_render = true;
        }

        // ── Keyboard shortcuts ────────────────────────────────────────
        let any_text_focused =
            ctx.memory(|m| m.focused()).is_some() && !response.has_focus();

        if !any_text_focused && (response.has_focus() || response.hovered()) {
            let (r_pressed, backspace_pressed, left, right, up, down, plus, minus) =
                ctx.input(|i| {
                    (
                        i.key_pressed(egui::Key::R),
                        i.key_pressed(egui::Key::Backspace),
                        i.key_pressed(egui::Key::ArrowLeft),
                        i.key_pressed(egui::Key::ArrowRight),
                        i.key_pressed(egui::Key::ArrowUp),
                        i.key_pressed(egui::Key::ArrowDown),
                        i.key_pressed(egui::Key::Plus)
                            || i.key_pressed(egui::Key::Equals),
                        i.key_pressed(egui::Key::Minus),
                    )
                });

            if r_pressed {
                self.params.set_from_default_bounds();
                self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
                self.history.clear();
                self.needs_render = true;
            }
            if backspace_pressed {
                if let Some(prev) = self.history.pop() {
                    prev.apply_to(&mut self.params);
                    self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
                    self.needs_render = true;
                }
            }

            // Arrow key panning (10% of view per press)
            let pan_frac: f64 = 0.1;
            if left || right || up || down {
                self.push_history();
                self.params.ensure_precision();
                self.mark_interactive();
            }
            if left {
                let dx = Float::with_val(
                    self.params.center_re.prec(),
                    -2.0 * self.params.half_range_x * pan_frac,
                );
                self.params.center_re += dx;
                self.needs_render = true;
            }
            if right {
                let dx = Float::with_val(
                    self.params.center_re.prec(),
                    2.0 * self.params.half_range_x * pan_frac,
                );
                self.params.center_re += dx;
                self.needs_render = true;
            }
            if up {
                let dy = Float::with_val(
                    self.params.center_im.prec(),
                    -2.0 * self.params.half_range_y * pan_frac,
                );
                self.params.center_im += dy;
                self.needs_render = true;
            }
            if down {
                let dy = Float::with_val(
                    self.params.center_im.prec(),
                    2.0 * self.params.half_range_y * pan_frac,
                );
                self.params.center_im += dy;
                self.needs_render = true;
            }

            // +/- keyboard zoom (centered)
            if plus || minus {
                let factor: f64 = if plus { 0.85 } else { 1.0 / 0.85 };
                self.push_history();
                self.params.half_range_x *= factor;
                self.params.half_range_y *= factor;
                self.params.ensure_precision();

                // Zoom preview centered at (0.5, 0.5)
                let f = factor as f32;
                let new_uv = egui::Rect::from_min_max(
                    egui::pos2(0.5 * (1.0 - f), 0.5 * (1.0 - f)),
                    egui::pos2(0.5 + 0.5 * f, 0.5 + 0.5 * f),
                );
                self.zoom_preview_uv = Some(match self.zoom_preview_uv {
                    Some(prev) => {
                        let pw = prev.width();
                        let ph = prev.height();
                        egui::Rect::from_min_max(
                            egui::pos2(
                                prev.min.x + new_uv.min.x * pw,
                                prev.min.y + new_uv.min.y * ph,
                            ),
                            egui::pos2(
                                prev.min.x + new_uv.max.x * pw,
                                prev.min.y + new_uv.max.y * ph,
                            ),
                        )
                    }
                    None => new_uv,
                });

                self.mark_interactive();
                self.needs_render = true;
            }
        }

        // Request focus on click so keyboard events work
        if response.clicked() {
            response.request_focus();
        }
    }
}
