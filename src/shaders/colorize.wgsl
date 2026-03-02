// Colorization compute shader.
// Reads iteration data + final z, produces RGBA output.
// Supports both escape-time and root-basin coloring.

struct Params {
    bounds: vec4<f32>,
    resolution: vec2<u32>,
    max_iter: u32,
    fractal_type: u32,
    julia_c: vec2<f32>,
    power: f32,
    relaxation: f32,
    color_mode: u32,     // 0 = escape-time, 1 = root-basin
    num_roots: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> iterations: array<f32>;
@group(0) @binding(2) var<storage, read> final_z: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read> roots: array<vec2<f32>>;

// ── HSV to RGB ───────────────────────────────────────────────────────────────

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let h6 = h * 6.0;
    let i = u32(floor(h6)) % 6u;
    let f = h6 - floor(h6);
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    switch i {
        case 0u: { return vec3<f32>(v, t, p); }
        case 1u: { return vec3<f32>(q, v, p); }
        case 2u: { return vec3<f32>(p, v, t); }
        case 3u: { return vec3<f32>(p, q, v); }
        case 4u: { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

// ── Color palettes ──────────────────────────────────────────────────────────

// Smooth escape-time coloring with a nice palette.
// Uses absolute iteration count so colors stay stable when max_iter changes.
fn escape_color(smooth_iter: f32, max_iter: f32) -> vec3<f32> {
    if smooth_iter >= max_iter {
        return vec3<f32>(0.0, 0.0, 0.0); // Set interior = black
    }

    // Absolute color mapping: hue cycles based on log of iteration count,
    // independent of max_iter. This keeps colors stable when changing iterations.
    let log_iter = log2(smooth_iter + 1.0);
    let hue = fract(log_iter * 0.15 + 0.6);

    // Saturation and value based on log iteration — no max_iter dependency.
    let sat = 0.7 + 0.3 * cos(log_iter * 0.5);
    let val = 0.85 + 0.15 * cos(log_iter * 0.7);

    return hsv_to_rgb(hue, sat, val);
}

// Root-basin coloring for Newton/Nova fractals.
fn basin_color(smooth_iter: f32, z: vec2<f32>, max_iter: f32, n_roots: u32) -> vec3<f32> {
    if smooth_iter >= max_iter {
        return vec3<f32>(0.0, 0.0, 0.0); // Unconverged = black
    }

    // Find nearest root
    var min_dist: f32 = 1e20;
    var root_id: u32 = 0u;
    for (var i: u32 = 0u; i < n_roots; i++) {
        let root = roots[i];
        let dx = z.x - root.x;
        let dy = z.y - root.y;
        let d = dx * dx + dy * dy;
        if d < min_dist {
            min_dist = d;
            root_id = i;
        }
    }

    // Golden angle spacing for root hues
    let golden = 0.618033988749895;
    let hue = fract(f32(root_id) * golden);

    // Shade by iteration count (faster = brighter)
    let shade = 1.0 / (1.0 + 0.05 * smooth_iter);
    let sat = 0.55 + 0.4 * shade;

    return hsv_to_rgb(hue, sat, shade);
}

// ── Pack RGBA ────────────────────────────────────────────────────────────────

fn pack_rgba(rgb: vec3<f32>) -> u32 {
    let r = u32(clamp(rgb.x * 255.0, 0.0, 255.0));
    let g = u32(clamp(rgb.y * 255.0, 0.0, 255.0));
    let b = u32(clamp(rgb.z * 255.0, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

// ── Main ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.resolution.x;
    let h = params.resolution.y;
    if x >= w || y >= h { return; }

    let idx = y * w + x;
    let smooth_iter = iterations[idx];
    let z = final_z[idx];
    let max_iter = f32(params.max_iter);

    var color: vec3<f32>;

    if params.color_mode == 0u {
        color = escape_color(smooth_iter, max_iter);
    } else {
        color = basin_color(smooth_iter, z, max_iter, params.num_roots);
    }

    // The output buffer is raw-copied into an Rgba8UnormSrgb texture.
    // copy_buffer_to_texture does NOT apply color conversion — bytes are
    // stored verbatim. The sRGB decode happens when egui samples the
    // texture. Our HSV palette produces sRGB-like perceptual values, so
    // we store them directly — the sRGB decode on sampling is close enough
    // to identity for visually-authored palettes.
    output[idx] = pack_rgba(color);
}
