// Finalize compute shader.
// For escape-time (color_mode 0): reads accumulated iteration data, computes
// color from the averaged smooth iteration count.
// For basin coloring (color_mode 1): reads accumulated weighted color, divides
// by total weight.
// Packs result to RGBA u32 for the output buffer / texture copy.

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
    color_mode: u32,
    num_roots: u32,
    sample_offset: vec2<f32>,
    sample_weight: f32,
    stride: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn pack_rgba(rgb: vec3<f32>) -> u32 {
    let r = u32(clamp(rgb.x * 255.0, 0.0, 255.0));
    let g = u32(clamp(rgb.y * 255.0, 0.0, 255.0));
    let b = u32(clamp(rgb.z * 255.0, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

// -- HSV to RGB (duplicated from colorize.wgsl — WGSL has no includes) --------

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

// -- Escape-time palette (duplicated from colorize.wgsl) ----------------------

fn escape_color(smooth_iter: f32) -> vec3<f32> {
    let log_iter = log2(smooth_iter + 1.0);
    let hue = fract(log_iter * 0.15 + 0.6);
    let sat = 0.7 + 0.3 * cos(log_iter * 0.5);
    let val = 0.85 + 0.15 * cos(log_iter * 0.7);

    return hsv_to_rgb(hue, sat, val);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.resolution.x;
    let h = params.resolution.y;
    if x >= w || y >= h { return; }

    let idx = y * params.stride + x;
    let a = accum[idx];
    var color: vec3<f32>;

    if params.color_mode == 0u {
        // Escape-time: accum = (sum_smooth_iter_weighted, escaped_weight, total_weight, 0)
        let escaped_wt = a.y;
        let total_wt = a.z;
        if escaped_wt > 0.0 {
            let avg_iter = a.x / escaped_wt;
            let escaped_color = escape_color(avg_iter);
            // Blend with black based on fraction of samples that escaped
            color = escaped_color * (escaped_wt / total_wt);
        } else {
            color = vec3<f32>(0.0, 0.0, 0.0);
        }
    } else {
        // Basin coloring: accum.xyz = sum of weighted RGB, accum.w = sum of weights
        let weight = a.w;
        if weight > 0.0 {
            color = clamp(a.xyz / weight, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
        } else {
            color = vec3<f32>(0.0, 0.0, 0.0);
        }
    }

    output[idx] = pack_rgba(color);
}
