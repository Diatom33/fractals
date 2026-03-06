// Median finalize shader.
// Reads all per-sample iteration values, finds the median of exterior samples,
// maps that single iteration count to a color, composites with interior coverage.

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
    sample_weight: f32,       // total_weight (set by host)
    stride: u32,
    palette: u32,
    sample_index: u32,
    num_samples: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> iterations: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

// -- Oklab to linear sRGB ---------------------------------------------------

fn oklab_to_linear_srgb(L: f32, a_ok: f32, b_ok: f32) -> vec3<f32> {
    let l_ = L + 0.3963377774 * a_ok + 0.2158037573 * b_ok;
    let m_ = L - 0.1055613458 * a_ok - 0.0638541728 * b_ok;
    let s_ = L - 0.0894841775 * a_ok - 1.2914855480 * b_ok;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    let r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

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

// -- Palettes (same as colorize.wgsl) ----------------------------------------

fn palette_classic(smooth_iter: f32) -> vec3<f32> {
    let log_iter = log2(smooth_iter + 1.0);
    let hue = fract(log_iter * 0.15 + 0.6);
    let sat = 0.7 + 0.3 * cos(log_iter * 0.5);
    let val = 0.85 + 0.15 * cos(log_iter * 0.7);
    return hsv_to_rgb(hue, sat, val);
}

fn palette_oklab(smooth_iter: f32) -> vec3<f32> {
    let log_iter = log2(smooth_iter + 1.0);
    let hue_angle = log_iter * 0.9 + 0.5;
    let linear = oklab_to_linear_srgb(0.75, 0.12 * cos(hue_angle), 0.12 * sin(hue_angle));
    return pow(linear, vec3<f32>(1.0 / 2.2));
}

fn palette_smooth(smooth_iter: f32) -> vec3<f32> {
    let t = log2(smooth_iter + 1.0) * 0.1;
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.00, 0.10, 0.20);
    return a + b * cos(6.28318 * (c * t + d));
}

fn palette_mono(smooth_iter: f32) -> vec3<f32> {
    let log_iter = log2(smooth_iter + 1.0);
    let val = 0.5 + 0.5 * cos(log_iter * 0.4 + 1.0);
    return vec3<f32>(val * 0.15, val * 0.3, val);
}

fn escape_color(smooth_iter: f32) -> vec3<f32> {
    switch params.palette {
        case 1u: { return palette_oklab(smooth_iter); }
        case 2u: { return palette_smooth(smooth_iter); }
        case 3u: { return palette_mono(smooth_iter); }
        default: { return palette_classic(smooth_iter); }
    }
}

fn pack_rgba(rgb: vec3<f32>) -> u32 {
    let r = u32(clamp(rgb.x * 255.0, 0.0, 255.0));
    let g = u32(clamp(rgb.y * 255.0, 0.0, 255.0));
    let b = u32(clamp(rgb.z * 255.0, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

// -- Main: median of exterior iterations, color, composite --------------------

// Max samples we support for median (limited by private array size in WGSL)
const MAX_SAMPLES: u32 = 64u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.resolution.x;
    let h = params.resolution.y;
    if x >= w || y >= h { return; }

    let idx = y * params.stride + x;
    let max_iter = f32(params.max_iter);
    let n = min(params.num_samples, MAX_SAMPLES);
    let pixels_per_sample = params.stride * h;

    // Collect exterior iteration values
    var ext_iters: array<f32, 64>;
    var ext_count: u32 = 0u;

    for (var s: u32 = 0u; s < n; s++) {
        let iter_val = iterations[s * pixels_per_sample + idx];
        if iter_val < max_iter {
            ext_iters[ext_count] = iter_val;
            ext_count++;
        }
    }

    // All interior — pure black
    if ext_count == 0u {
        output[idx] = pack_rgba(vec3<f32>(0.0));
        return;
    }

    // Insertion sort (fine for N ≤ 64)
    for (var i: u32 = 1u; i < ext_count; i++) {
        let key = ext_iters[i];
        var j: i32 = i32(i) - 1;
        loop {
            if j < 0 { break; }
            if ext_iters[u32(j)] <= key { break; }
            ext_iters[u32(j + 1)] = ext_iters[u32(j)];
            j--;
        }
        ext_iters[u32(j + 1)] = key;
    }

    // Median
    var median_iter: f32;
    if ext_count % 2u == 1u {
        median_iter = ext_iters[ext_count / 2u];
    } else {
        median_iter = (ext_iters[ext_count / 2u - 1u] + ext_iters[ext_count / 2u]) * 0.5;
    }

    // Color the median iteration value
    let color = escape_color(median_iter);

    // Composite with interior coverage
    let ext_coverage = f32(ext_count) / f32(n);
    let final_color = color * ext_coverage;

    output[idx] = pack_rgba(final_color);
}
