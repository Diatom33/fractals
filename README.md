# Fractal Explorer

GPU-accelerated fractal renderer with an interactive GUI. Built with Rust, wgpu compute shaders (WGSL), and egui.

## Fractal Types

| Type | Formula | Description |
|------|---------|-------------|
| **Mandelbrot** | z = z^2 + c | Classic escape-time fractal. c = pixel, z0 = 0 |
| **Julia** | z = z^2 + c | Julia set. z0 = pixel, c = adjustable constant |
| **Burning Ship** | z = (\|Re(z)\| + i\|Im(z)\|)^2 + c | Non-analytic variant with absolute value before squaring |
| **Multibrot** | z = z^d + c | Generalized Mandelbrot with configurable power d |
| **Newton** | z = z - a*f(z)/f'(z) | Newton's method for z^n - 1, colored by basin of attraction |
| **Nova Julia** | z = z - a*f(z)/f'(z) + c | Nova fractal with fixed c, z0 = pixel |
| **Nova Mandelbrot** | z = z - a*f(z)/f'(z) + c | Nova fractal with c = pixel, z0 = critical point |

## Requirements

- Rust 1.75+ (`rustup` recommended)
- GPU with Vulkan, Metal, or DX12 support
- Linux: `libwayland-dev` and/or X11 dev libraries

## Build & Run

```bash
cargo run --release
```

## CLI Export

```bash
# Export a specific fractal type to PNG (1920x1080)
cargo run --release -- --export output.png --type mandelbrot
cargo run --release -- --export julia.png --type julia
cargo run --release -- --export ship.png --type burningship
cargo run --release -- --export newton.png --type newton
cargo run --release -- --export nova.png --type novamandelbrot
```

Available `--type` values: `mandelbrot`, `julia`, `burningship`, `multibrot`, `newton`, `novajulia`, `novamandelbrot`

## Controls

### Mouse
- **Scroll**: Zoom toward cursor
- **Click + Drag**: Pan
- **Double-click**: Reset view

### Keyboard
- **R**: Reset view to defaults
- **Backspace**: Undo last navigation
- **Arrow keys**: Pan 10% of view
- **+/-**: Zoom in/out (centered)

### Side Panel
- **Fractal Type**: Dropdown selector — switching types resets bounds to appropriate defaults
- **Max Iterations**: 10–2000 (logarithmic slider)
- **Power d**: Multibrot exponent (2.0–8.0)
- **Julia c**: Real and imaginary sliders (-2.0 to 2.0)
- **Degree n**: Newton/Nova polynomial degree for z^n - 1 (2–8)
- **Relaxation a**: Nova damping parameter (0.1–2.0)
- **Export**: Save current view to PNG

Controls are shown/hidden based on the selected fractal type.

## Architecture

```
src/
├── main.rs          # Entry point + CLI export mode
├── app.rs           # FractalApp: egui UI, mouse/keyboard handling
├── gpu.rs           # GpuState: wgpu device, pipelines, buffers, textures
├── fractals.rs      # FractalType enum, FractalParams, GpuParams uniform struct
└── shaders/
    ├── escape.wgsl  # Mandelbrot / Julia / Burning Ship / Multibrot iteration
    ├── newton.wgsl  # Newton / Nova Julia / Nova Mandelbrot iteration
    └── colorize.wgsl # Escape-time smooth coloring + root-basin coloring
```

### GPU Pipeline

Two-pass compute shader pipeline:

1. **Iterate** (`escape.wgsl` or `newton.wgsl`): Runs the fractal iteration for every pixel. Outputs smooth iteration count (`f32`) and final z value (`vec2<f32>`) per pixel.
2. **Colorize** (`colorize.wgsl`): Reads iteration data and produces RGBA pixels. Escape-time fractals use smooth coloring (`iter + 1 - log2(log2(|z|)) / log2(d)`). Newton/Nova fractals use root-basin coloring with golden-angle hue spacing.

The output buffer is copied to a wgpu texture, registered with egui for display.

### Key Design Decisions

- **Complex numbers in WGSL**: `vec2<f32>` where x = real, y = imaginary. Custom `cmul`, `csqr`, `cdiv`, `cpow_int` functions.
- **Newton roots**: Analytically computed as nth roots of unity for z^n - 1 (no GPU readback or clustering needed).
- **Convergence criterion**: Escape-time checks `|z|^2 > 256`. Newton/Nova checks `|z_new - z|^2 < tol^2` (step size, not residual — critical for Nova where f(z) != 0 at fixed points).
- **Width alignment**: Buffer widths are rounded up to multiples of 64 pixels to satisfy wgpu's `COPY_BYTES_PER_ROW_ALIGNMENT` (256 bytes).
- **Render-on-demand**: Only re-renders when parameters change (dirty flag + params hash comparison).

