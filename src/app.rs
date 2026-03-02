/// Fractal Explorer egui application.

use crate::fractals::{FractalParams, FractalType};
use crate::gpu::GpuState;

pub struct FractalApp {
    gpu: Option<GpuState>,
    params: FractalParams,
    prev_params_hash: u64,
    history: Vec<[f32; 4]>, // bounds history for undo
    needs_render: bool,

    // Drag state
    drag_start: Option<egui::Pos2>,
    drag_bounds: Option<[f32; 4]>,

    // Export
    export_path: String,
    export_msg: String,
    export_msg_is_error: bool,

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
            drag_start: None,
            drag_bounds: None,
            export_path: String::from("fractal.png"),
            export_msg: String::new(),
            export_msg_is_error: false,
            cursor_complex: None,
            last_display_w: 0,
            last_display_h: 0,
        }
    }

    /// Adjust bounds so that the scale (units per pixel) is equal in x and y,
    /// keeping the center of the view fixed. The larger scale wins so nothing
    /// gets cropped — the smaller dimension is expanded to match.
    /// Only for user-initiated changes (type switch, reset, undo) where we
    /// need to establish a correct aspect ratio from scratch.
    fn correct_aspect_ratio(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let [x_min, x_max, y_min, y_max] = self.params.bounds;
        let cx = (x_min + x_max) * 0.5;
        let cy = (y_min + y_max) * 0.5;

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        let scale_x = x_range / width as f32;
        let scale_y = y_range / height as f32;

        // Use the larger scale so the view expands rather than crops
        let scale = scale_x.max(scale_y);

        let new_x_range = scale * width as f32;
        let new_y_range = scale * height as f32;

        self.params.bounds = [
            cx - new_x_range * 0.5,
            cx + new_x_range * 0.5,
            cy - new_y_range * 0.5,
            cy + new_y_range * 0.5,
        ];

        self.last_display_w = width;
        self.last_display_h = height;
    }

    /// Scale bounds proportionally when the display/GPU dimensions change.
    /// Preserves center and per-pixel scale in each axis independently.
    /// Unlike correct_aspect_ratio, this never uses max() so it can't ratchet.
    fn scale_bounds_to_new_size(&mut self, new_w: u32, new_h: u32) {
        if self.last_display_w == 0 || self.last_display_h == 0 || new_w == 0 || new_h == 0 {
            return;
        }
        let [x_min, x_max, y_min, y_max] = self.params.bounds;
        let cx = (x_min + x_max) * 0.5;
        let cy = (y_min + y_max) * 0.5;

        let x_range = (x_max - x_min) * new_w as f32 / self.last_display_w as f32;
        let y_range = (y_max - y_min) * new_h as f32 / self.last_display_h as f32;

        self.params.bounds = [
            cx - x_range * 0.5,
            cx + x_range * 0.5,
            cy - y_range * 0.5,
            cy + y_range * 0.5,
        ];

        self.last_display_w = new_w;
        self.last_display_h = new_h;
    }

    fn params_hash(&self) -> u64 {
        let (dw, h, stride, ss) = self.gpu.as_ref().map_or((960, 720, 960, 1), |g| {
            (g.display_width, g.height, g.width, g.supersampling)
        });
        let binding = self.params.to_gpu_params(dw, h, stride);
        // XOR in supersampling so SS changes trigger re-render
        let ss_extra = (ss as u64) << 48;
        let bytes = bytemuck::bytes_of(&binding);
        let mut hash = 0u64;
        for chunk in bytes.chunks(8) {
            let mut arr = [0u8; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            hash ^= u64::from_le_bytes(arr);
        }
        hash ^ ss_extra
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
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // ── Ctrl+Q to quit ───────────────────────────────────────────────
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Q)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        let render_state = frame.wgpu_render_state();

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
                            self.params.bounds = self.params.fractal_type.default_bounds();
                            self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
                            self.history.clear();
                            self.needs_render = true;
                        }

                        // ── Parameters ────────────────────────────────
                        section_header(ui, "Parameters");

                        ui.label("Max iterations:");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.params.max_iter, 10..=2000)
                                    .logarithmic(true),
                            )
                            .changed()
                        {
                            self.needs_render = true;
                        }

                        ui.add_space(2.0);
                        ui.label("Supersampling:");
                        let ss_label = match self.params.supersampling {
                            2 => "2x2 (4 samples)",
                            3 => "3x3 (9 samples)",
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
                                    "2x2 (4 samples)",
                                );
                                ui.selectable_value(
                                    &mut self.params.supersampling,
                                    3,
                                    "3x3 (9 samples)",
                                );
                            });
                        if self.params.supersampling != prev_ss {
                            self.needs_render = true;
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

                        // ── Navigation ────────────────────────────────
                        section_header(ui, "Navigation");

                        ui.horizontal(|ui| {
                            if ui.button("Reset View").clicked() {
                                self.params.bounds = self.params.fractal_type.default_bounds();
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
                                    self.params.bounds = prev;
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

                        // Show current view bounds
                        let [x_min, x_max, y_min, y_max] = self.params.bounds;
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
                        if gpu.supersampling > 1 {
                            ui.monospace(format!("{}x SS", gpu.supersampling));
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
                        let [xn, xx, yn, yx] = self.params.bounds;
                        let asp = ((xx - xn) / (yx - yn)) / (gpu.display_width as f32 / gpu.height as f32);
                        ui.monospace(
                            egui::RichText::new(format!(
                                "asp={:.3}",
                                asp,
                            )).weak(),
                        );
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
                let display_w = (available.x as u32).max(64);
                let display_h = (available.y as u32).max(64);

                // Resize GPU buffers/texture if needed
                let mut new_display_dims: Option<(u32, u32)> = None;
                if let Some(gpu) = &mut self.gpu {
                    let ss = self.params.supersampling;
                    let size_changed = gpu.display_width != display_w || gpu.height != display_h;

                    if size_changed || gpu.supersampling != ss {
                        gpu.resize(display_w, display_h, ss);
                        self.needs_render = true;
                    }
                    if size_changed {
                        new_display_dims = Some((gpu.display_width, gpu.height));
                    }
                }
                if let Some((new_w, new_h)) = new_display_dims {
                    if self.last_display_w > 0 {
                        // Window resize: scale bounds proportionally to display dimensions.
                        // No max() → no ratchet from sub-pixel display jitter.
                        self.scale_bounds_to_new_size(new_w, new_h);
                    } else {
                        // First frame: establish correct aspect ratio from scratch
                        self.correct_aspect_ratio(new_w, new_h);
                    }
                }

                // Render if needed (params changed or flagged)
                let hash = if self.needs_render { 0 } else { self.params_hash() };
                let did_render = self.needs_render || hash != self.prev_params_hash;
                if did_render {
                    if let Some(gpu) = &mut self.gpu {
                        gpu.render(&self.params);
                    }
                    self.prev_params_hash = self.params_hash();
                    self.needs_render = false;
                }

                // Register/update texture with egui-wgpu renderer
                if let (Some(gpu), Some(rs)) = (&mut self.gpu, render_state) {
                    let mut renderer = rs.renderer.write();
                    let was_unregistered = gpu.texture_id.is_none();
                    gpu.ensure_texture_registered(&mut renderer);
                    if did_render || was_unregistered {
                        gpu.update_texture(&mut renderer);
                    }
                }

                // Display image filling the panel
                if let Some(gpu) = &self.gpu {
                    if let Some(tex_id) = gpu.texture_id {
                        let size = egui::vec2(available.x, available.y);
                        let (response, _painter) =
                            ui.allocate_painter(size, egui::Sense::click_and_drag());
                        let rect = response.rect;

                        ui.painter().image(
                            tex_id,
                            rect,
                            egui::Rect::from_min_max(
                                egui::pos2(0.0, 0.0),
                                egui::pos2(1.0, 1.0),
                            ),
                            egui::Color32::WHITE,
                        );

                        // One-shot diagnostic: print all display dimensions
                        if did_render && gpu.last_render_ms > 0.0 {
                            static ONCE: std::sync::Once = std::sync::Once::new();
                            ONCE.call_once(|| {
                                eprintln!(
                                    "DIAG: available=({:.1}, {:.1}) rect=({:.1}x{:.1}) display={}x{} stride={} tex={}x{}",
                                    available.x, available.y,
                                    rect.width(), rect.height(),
                                    gpu.display_width, gpu.height,
                                    gpu.width,
                                    gpu.texture.size().width, gpu.texture.size().height,
                                );
                            });
                        }

                        // Update cursor complex coordinates for side panel display
                        if let Some(pos) = response.hover_pos() {
                            let frac_x =
                                ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
                            let frac_y =
                                ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0);
                            let [bx_min, bx_max, by_min, by_max] = self.params.bounds;
                            let cx = bx_min as f64
                                + frac_x as f64 * (bx_max - bx_min) as f64;
                            let cy = by_min as f64
                                + frac_y as f64 * (by_max - by_min) as f64;
                            self.cursor_complex = Some((cx, cy));
                        } else {
                            self.cursor_complex = None;
                        }

                        // Mouse & keyboard interaction
                        self.handle_input(ctx, &response, rect);
                    }
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
        let [x_min, x_max, y_min, y_max] = self.params.bounds;

        // ── Scroll zoom toward cursor ─────────────────────────────────
        if response.hovered() {
            let scroll = ctx.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                if let Some(pos) = response.hover_pos() {
                    let frac_x = (pos.x - rect.min.x) / rect.width();
                    let frac_y = (pos.y - rect.min.y) / rect.height();
                    let cx = x_min + frac_x * (x_max - x_min);
                    let cy = y_min + frac_y * (y_max - y_min);

                    let factor = if scroll > 0.0 { 0.85 } else { 1.0 / 0.85 };
                    self.history.push(self.params.bounds);
                    self.params.bounds = [
                        cx - (cx - x_min) * factor,
                        cx + (x_max - cx) * factor,
                        cy - (cy - y_min) * factor,
                        cy + (y_max - cy) * factor,
                    ];
                    self.needs_render = true;
                }
            }
        }

        // ── Drag to pan ───────────────────────────────────────────────
        if response.drag_started() {
            self.drag_start = response.interact_pointer_pos();
            self.drag_bounds = Some(self.params.bounds);
        }
        if response.dragged() {
            if let (Some(start), Some(orig_bounds)) = (self.drag_start, self.drag_bounds) {
                if let Some(current) = response.interact_pointer_pos() {
                    let dx_px = current.x - start.x;
                    let dy_px = current.y - start.y;
                    let dx = -dx_px / rect.width() * (orig_bounds[1] - orig_bounds[0]);
                    let dy = -dy_px / rect.height() * (orig_bounds[3] - orig_bounds[2]);
                    self.params.bounds = [
                        orig_bounds[0] + dx,
                        orig_bounds[1] + dx,
                        orig_bounds[2] + dy,
                        orig_bounds[3] + dy,
                    ];
                    self.needs_render = true;
                }
            }
        }
        if response.drag_stopped() {
            if let Some(orig) = self.drag_bounds.take() {
                if orig != self.params.bounds {
                    self.history.push(orig);
                }
            }
            self.drag_start = None;
        }

        // ── Double-click to reset ─────────────────────────────────────
        if response.double_clicked() {
            self.params.bounds = self.params.fractal_type.default_bounds();
            self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
            self.history.clear();
            self.needs_render = true;
        }

        // ── Keyboard shortcuts ────────────────────────────────────────
        // Only process when no text edit widget has focus. We check if
        // something else has focus that is not our image response.
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
                self.params.bounds = self.params.fractal_type.default_bounds();
                self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
                self.history.clear();
                self.needs_render = true;
            }
            if backspace_pressed {
                if let Some(prev) = self.history.pop() {
                    self.params.bounds = prev;
                    self.correct_aspect_ratio(self.last_display_w, self.last_display_h);
                    self.needs_render = true;
                }
            }

            // Arrow key panning (10% of view per press)
            let pan_frac = 0.1;
            let dx = (x_max - x_min) * pan_frac;
            let dy = (y_max - y_min) * pan_frac;
            if left || right || up || down {
                self.history.push(self.params.bounds);
            }
            if left {
                self.params.bounds[0] -= dx;
                self.params.bounds[1] -= dx;
                self.needs_render = true;
            }
            if right {
                self.params.bounds[0] += dx;
                self.params.bounds[1] += dx;
                self.needs_render = true;
            }
            if up {
                self.params.bounds[2] -= dy;
                self.params.bounds[3] -= dy;
                self.needs_render = true;
            }
            if down {
                self.params.bounds[2] += dy;
                self.params.bounds[3] += dy;
                self.needs_render = true;
            }

            // +/- keyboard zoom (centered)
            if plus || minus {
                let factor = if plus { 0.85 } else { 1.0 / 0.85 };
                let cx = (x_min + x_max) * 0.5;
                let cy = (y_min + y_max) * 0.5;
                self.history.push(self.params.bounds);
                self.params.bounds = [
                    cx - (cx - x_min) * factor,
                    cx + (x_max - cx) * factor,
                    cy - (cy - y_min) * factor,
                    cy + (y_max - cy) * factor,
                ];
                self.needs_render = true;
            }
        }

        // Request focus on click so keyboard events work
        if response.clicked() {
            response.request_focus();
        }
    }
}
