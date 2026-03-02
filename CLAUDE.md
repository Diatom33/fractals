# Fractal Explorer — Project Guide

## Quick Reference

- **Language**: Rust (edition 2021)
- **GPU**: wgpu 24 compute shaders in WGSL
- **GUI**: egui 0.31 via eframe (wgpu backend, x11 + wayland features)
- **Build**: `cargo run --release` (always use release — debug is unusably slow for GPU)
- **Export CLI**: `cargo run --release -- --export file.png --type mandelbrot`
- **Keyboard**: Ctrl+Q to close, R/Backspace to reset view, scroll to zoom, double-click to center

## File Map

| File | Owner | Purpose |
|------|-------|---------|
| `src/main.rs` | shared | Entry point, CLI export mode |
| `src/app.rs` | UI code | `FractalApp` struct, egui layout, mouse/keyboard input |
| `src/gpu.rs` | GPU code | `GpuState`: wgpu device, pipelines, buffers, render dispatch |
| `src/fractals.rs` | data | `FractalType` enum (7 types), `FractalParams`, `GpuParams` uniform |
| `src/shaders/escape.wgsl` | shader | Mandelbrot, Julia, Burning Ship, Multibrot iteration |
| `src/shaders/newton.wgsl` | shader | Newton, Nova Julia, Nova Mandelbrot iteration |
| `src/shaders/colorize.wgsl` | shader | HSV coloring: escape-time smooth + root-basin modes |
| `src/shaders/finalize.wgsl` | shader | Divides accumulated samples by weight, packs to RGBA |


## Critical Gotchas

### WGSL Reserved Keywords
`smooth`, `step`, `sample`, `distance`, `length`, `normalize` are all reserved in WGSL. Use `smooth_val`, `step_val`, etc. instead.

### Nova Convergence
Newton/Nova convergence MUST check `|z_new - z|^2 < tol^2` (step size), NOT `|f(z)|^2 < tol^2` (residual). Nova fixed points satisfy `a*f(z)/f'(z) = c`, so f(z) != 0 at convergence when c != 0. This was a bug caught twice — once in Python, once in Rust.

### Buffer Width Alignment
wgpu's `copy_buffer_to_texture` requires `bytes_per_row` to be a multiple of 256. Since we use RGBA8 (4 bytes/pixel), width must be a multiple of 64. The `align_width()` function in `gpu.rs` handles this. Always use it when setting buffer dimensions.

### vec2 * vec2 in WGSL
`vec2<f32> * vec2<f32>` is component-wise multiplication, NOT scalar broadcast. To multiply a complex number by a scalar, use `scalar * vec` (f32 * vec2), not `vec2(scalar, 0.0) * vec`. The latter zeros the imaginary part.

### GpuParams Struct Alignment
`GpuParams` in `fractals.rs` must match the WGSL `Params` struct exactly — same field order, same sizes, padded to 80 bytes. It's `#[repr(C)]` with `bytemuck::Pod + Zeroable`. If you add a field, add it in both places and adjust `_pad`. The Params struct must be updated in ALL FOUR `.wgsl` files (escape, newton, colorize, finalize).

### Texture Registration
The egui texture must be registered once via `renderer.register_native_texture()` and updated via `renderer.update_egui_texture_from_wgpu_texture()` after each render. The `texture_id` is `None` until first registration and must be re-registered after resize (since the texture is recreated).

## Architecture Overview

```
User input → FractalParams changed → dirty flag set
                                          ↓
                              GpuState::render()
                                    ↓
                        [write roots buffer if Newton/Nova]
                        [clear accumulation buffer]
                                    ↓
                    ┌─── for each sub-pixel sample (ss×ss) ───┐
                    │                                          │
                    │  [write params with sample_offset]       │
                    │                                          │
                    │  Pass 1: escape.wgsl OR newton.wgsl      │
                    │      → iterations buffer (f32/pixel)     │
                    │      → final_z buffer (vec2<f32>/pixel)  │
                    │                                          │
                    │  Pass 2: colorize.wgsl                   │
                    │      → accumulate weighted color into    │
                    │        accum buffer (vec4<f32>/pixel)    │
                    └──────────────────────────────────────────┘
                                    ↓
                        Pass 3: finalize.wgsl
                            → divide by total weight, pack RGBA
                            → output buffer (u32/pixel)
                                    ↓
                        copy_buffer_to_texture → display texture
                                    ↓
                        egui paints texture to central panel
```

### Supersampling Architecture
Multi-pass accumulation: each sub-pixel sample is a full render pass with a sub-pixel offset applied in complex plane coordinate mapping. All buffers stay at output resolution (no Nx inflation), avoiding the 128MB `max_storage_buffer_binding_size` limit. The colorize shader accumulates weighted RGB into a vec4<f32> buffer; the finalize shader divides by total weight and packs to RGBA.

## Adding a New Fractal Type

1. Add variant to `FractalType` enum in `fractals.rs`
2. Implement `name()`, `shader_index()`, `is_escape_time()`, `needs_roots()`, `default_bounds()`, `visible_controls()` for it
3. If escape-time: add a case in `escape.wgsl` (branch on `params.fractal_type`)
4. If convergence-based: add a case in `newton.wgsl`
5. Coloring is automatic (escape-time or basin based on `color_mode`)
6. Add to `FractalType::ALL` array
7. The UI dropdown and CLI parser pick it up automatically

## Adding a New Parameter

1. Add field to `FractalParams` in `fractals.rs`
2. Add field to `GpuParams` (must maintain alignment — adjust `_pad`)
3. Add matching field to `Params` struct in ALL four `.wgsl` files (escape, newton, colorize, finalize)
4. Add UI control in `app.rs` (gated by `visible_controls()`)
5. Wire it through `to_gpu_params()` in `fractals.rs`

## Performance Notes

- RTX 4090 renders 1080p Mandelbrot in ~2-5ms
- Newton is slower due to complex division and power operations (~5-15ms)
- `device.poll(Maintain::Wait)` blocks until GPU finishes — acceptable for on-demand rendering
- Only re-render when params hash changes or `needs_render` flag is set
- Texture updates only happen after actual renders, not every egui frame

## Dependencies

Key crates and why:
- `eframe` 0.31: egui desktop app framework with wgpu backend
- `wgpu` 24: GPU compute (matched to eframe's internal wgpu version)
- `bytemuck` 1: Safe casting of `GpuParams` to bytes for uniform buffer upload
- `pollster` 0.4: Block on async wgpu calls in CLI export mode
- `image` 0.25: PNG encoding for export
