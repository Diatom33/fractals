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
        }
    }

    fn params_hash(&self) -> u64 {
        let binding = self.params.to_gpu_params(
            self.gpu.as_ref().map_or(960, |g| g.width),
            self.gpu.as_ref().map_or(720, |g| g.height),
        );
        let bytes = bytemuck::bytes_of(&binding);
        let mut hash = 0u64;
        for chunk in bytes.chunks(8) {
            let mut arr = [0u8; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            hash ^= u64::from_le_bytes(arr);
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
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
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
                        ui.monospace(format!("{}x{}", gpu.width, gpu.height));
                        ui.separator();
                        ui.monospace(format!("{:.1} ms", gpu.last_render_ms));
                        ui.separator();
                        ui.monospace(egui::RichText::new(&gpu.gpu_name).weak());
                        ui.separator();
                        ui.monospace(
                            egui::RichText::new(format!("{} iters", self.params.max_iter)).weak(),
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
                let img_w = (available.x as u32).max(64);
                let img_h = (available.y as u32).max(64);

                // Resize GPU texture if panel size changed
                if let Some(gpu) = &mut self.gpu {
                    if gpu.width != img_w || gpu.height != img_h {
                        gpu.resize(img_w, img_h);
                        self.needs_render = true;
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
                self.history.clear();
                self.needs_render = true;
            }
            if backspace_pressed {
                if let Some(prev) = self.history.pop() {
                    self.params.bounds = prev;
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
