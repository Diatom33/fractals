// Colorization compute shader.
// Reads iteration data + final z, produces weighted color into accumulation buffer.
// Supports both escape-time and root-basin coloring.
// Called once per sub-pixel sample; accumulates into a vec4<f32> buffer (rgb + weight).

struct Params {
    center_hi: vec2<f32>,
    center_lo: vec2<f32>,
    pixel_step: vec2<f32>,
    resolution: vec2<u32>,
    max_iter: u32,
    fractal_type: u32,
    julia_c: vec2<f32>,
    power: f32,
    relaxation: f32,
    color_mode: u32,         // 0 = escape-time, 1 = root-basin
    num_roots: u32,
    sample_offset: vec2<f32>,
    sample_weight: f32,
    stride: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> iterations: array<f32>;
@group(0) @binding(2) var<storage, read> final_z: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> roots: array<vec2<f32>>;

// -- HSV to RGB ---------------------------------------------------------------

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

// -- Color palettes -----------------------------------------------------------

// Smooth escape-time coloring with a nice palette.
fn escape_color(smooth_iter: f32, max_iter: f32) -> vec3<f32> {
    if smooth_iter >= max_iter {
        return vec3<f32>(0.0, 0.0, 0.0); // Set interior = black
    }

    let log_iter = log2(smooth_iter + 1.0);
    let hue = fract(log_iter * 0.15 + 0.6);
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

// -- Colorize a single sample -------------------------------------------------

fn colorize_sample(smooth_iter: f32, z: vec2<f32>, max_iter: f32) -> vec3<f32> {
    if params.color_mode == 0u {
        return escape_color(smooth_iter, max_iter);
    } else {
        return basin_color(smooth_iter, z, max_iter, params.num_roots);
    }
}

// -- Main ---------------------------------------------------------------------
// Dispatch covers the OUTPUT resolution (width, height).
// Each invocation colorizes one sample and accumulates (adds) the weighted
// color into the accum buffer. The finalize shader divides by total weight.

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.resolution.x;
    let h = params.resolution.y;
    let x = gid.x;
    let y = gid.y;
    if x >= w || y >= h { return; }

    let idx = y * params.stride + x;
    let max_iter = f32(params.max_iter);
    let smooth_iter = iterations[idx];
    let z = final_z[idx];
    let wt = params.sample_weight;
    let prev = accum[idx];

    if params.color_mode == 0u {
        // Escape-time: accumulate raw iteration data instead of colors.
        // accum = (sum_smooth_iter_weighted, escaped_weight, total_weight, 0)
        if smooth_iter < max_iter {
            accum[idx] = prev + vec4<f32>(smooth_iter * wt, wt, wt, 0.0);
        } else {
            accum[idx] = prev + vec4<f32>(0.0, 0.0, wt, 0.0);
        }
    } else {
        // Basin coloring: accumulate weighted color as before.
        // accum.xyz = sum of weighted RGB, accum.w = sum of weights
        let color = basin_color(smooth_iter, z, max_iter, params.num_roots);
        accum[idx] = prev + vec4<f32>(color * wt, wt);
    }
}
