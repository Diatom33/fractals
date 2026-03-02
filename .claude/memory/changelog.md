# Changes Tried (Chronological)

## 1. CPU Readback Pipeline (replacing native texture interop)
**Hypothesis**: The native wgpu texture → egui interop was causing non-square pixels and broken supersampling.
**What was done**:
- Removed `texture`, `texture_view`, `texture_id` from `GpuState`
- Added `readback_buffer` with `MAP_READ` usage
- Changed `render()` to use `copy_buffer_to_buffer` instead of `copy_buffer_to_texture`
- Added `read_pixels()` method for CPU readback
- In `app.rs`: replaced native texture registration with `egui::ColorImage` + `ctx.load_texture()`
**Result**: Built and ran, CLI export worked, but **fixed neither problem** — pixels were still rectangles, supersampling still had no visible effect.

## 2. pixels_per_point investigation
**Hypothesis**: egui's DPI scaling (`pixels_per_point`) was making logical vs physical pixel mismatch → non-square pixels.
**What was done**: Added a one-shot diagnostic printing `ppp`, available size, and physical dimensions.
**Result**: `ppp=1.0` on the user's system. **Ruled out** as the cause.

## 3. Mitchell-Netravali filter weight fix
**Hypothesis**: Supersampling wasn't visibly working because the filter weights were wrong.
**What was found**: `mitchell_1d(offset_x / radius * 2.0)` with `radius=0.75` mapped offsets of ±0.5 to ±1.33 in filter domain — deep in the near-zero negative lobe (~0.001 weight).
**What was done**: Changed to `mitchell_1d(offset_x)` so offsets are evaluated directly in pixel units.
**Result**: Weights are now meaningful. **Fixed supersampling**.

## 4. f32 → f64 bounds on CPU side
**Hypothesis**: The block artifacts at deep zoom (visible in user's screenshot near x≈-2.0) were caused by f32 precision loss — per-pixel step smaller than f32 ULP → groups of pixels mapping to same coordinate.
**What was done**:
- `FractalParams.bounds`: `[f32; 4]` → `[f64; 4]`
- `default_bounds()` returns `[f64; 4]`
- `app.rs`: `history`, `drag_bounds`, all zoom/pan/aspect-ratio arithmetic converted to f64
- `main.rs`: `--bounds` parsed as f64
**Result**: CPU-side precision loss during zoom accumulation eliminated.

## 5. Double-single (emulated f64) in GPU shaders
**Hypothesis**: Even with f64 bounds on CPU, the shader still needs better-than-f32 precision for coordinate mapping at deep zoom.
**What was done**:
- `GpuParams` layout changed: replaced `bounds: [f32; 4]` with `center_hi: [f32; 2]`, `center_lo: [f32; 2]`, `pixel_step: [f32; 2]` (Dekker splitting in `to_gpu_params()`)
- `escape.wgsl`: Full rewrite with double-single arithmetic primitives (`two_sum`, `two_prod`, `ds_add`, `ds_mul`), coordinate mapping uses `two_prod` + `ds_add` for ~48-bit mantissa
- `newton.wgsl`: Updated Params struct, uses center+step (f32 iteration — complex division too involved for double-single)
- `colorize.wgsl`, `finalize.wgsl`: Updated Params struct
**Result**: Deep zoom near x≈-1.78 produces smooth pixels in CLI export — **no block artifacts**. All 4 WGSL shaders updated, all fractal types export correctly.

## Current State
Everything builds, CLI exports work for all fractal types including deep zoom with supersampling. Interactive app needs testing to confirm zoom/pan/drag all work with f64 bounds.

## Original Problems
1. **Non-square pixels / block artifacts at deep zoom** — Fixed by #4 + #5 (f64 CPU bounds + double-single GPU coords)
2. **Supersampling had no visible effect** — Fixed by #3 (Mitchell filter weight scaling)
