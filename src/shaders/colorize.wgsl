// Colorization compute shader.
// Reads iteration data + final z, produces weighted color into accumulation buffer.
// Supports both escape-time and root-basin coloring with multiple palettes.
// Called once per sub-pixel sample; accumulates into a vec4<f32> buffer (rgb + weight).
// Both color modes use color-domain averaging: compute RGB per sample, accumulate weighted.

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
    palette: u32,            // 0=Classic, 1=Oklab, 2=Smooth, 3=Monochrome
    _pad: u32,
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

// -- Oklab to linear sRGB -----------------------------------------------------

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

// -- Escape-time palettes -----------------------------------------------------

// Palette 0: Classic HSV oscillating hue
fn palette_classic(smooth_iter: f32) -> vec3<f32> {
    let log_iter = log2(smooth_iter + 1.0);
    let hue = fract(log_iter * 0.15 + 0.6);
    let sat = 0.7 + 0.3 * cos(log_iter * 0.5);
    let val = 0.85 + 0.15 * cos(log_iter * 0.7);
    return hsv_to_rgb(hue, sat, val);
}

// Palette 1: Oklab perceptually uniform — constant lightness, varying hue
fn palette_oklab(smooth_iter: f32) -> vec3<f32> {
    let log_iter = log2(smooth_iter + 1.0);
    let hue_angle = log_iter * 0.9 + 0.5;  // radians, cycles through hues
    let L = 0.75;                            // constant perceptual lightness
    let C = 0.12;                            // chroma (saturation)
    let a_ok = C * cos(hue_angle);
    let b_ok = C * sin(hue_angle);
    // oklab_to_linear_srgb returns linear RGB — we convert to sRGB for display
    let linear = oklab_to_linear_srgb(L, a_ok, b_ok);
    return pow(linear, vec3<f32>(1.0 / 2.2));
}

// Palette 2: Smooth iq-style cosine gradient
fn palette_smooth(smooth_iter: f32) -> vec3<f32> {
    let t = log2(smooth_iter + 1.0) * 0.1;
    // iq's cosine palette: color = a + b * cos(2π(c*t + d))
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.00, 0.10, 0.20);
    return a + b * cos(6.28318 * (c * t + d));
}

// Palette 3: Monochrome — single cool hue, varying brightness
fn palette_mono(smooth_iter: f32) -> vec3<f32> {
    let log_iter = log2(smooth_iter + 1.0);
    let t = fract(log_iter * 0.12);
    // Smooth oscillation in blue-cyan range
    let val = 0.5 + 0.5 * cos(log_iter * 0.4 + 1.0);
    return vec3<f32>(val * 0.15, val * 0.3, val);
}

// Dispatch to selected palette
fn escape_color(smooth_iter: f32, max_iter: f32) -> vec3<f32> {
    if smooth_iter >= max_iter {
        return vec3<f32>(0.0, 0.0, 0.0); // Set interior = black
    }

    switch params.palette {
        case 1u: { return palette_oklab(smooth_iter); }
        case 2u: { return palette_smooth(smooth_iter); }
        case 3u: { return palette_mono(smooth_iter); }
        default: { return palette_classic(smooth_iter); }
    }
}

// -- Root-basin coloring (Newton/Nova) ----------------------------------------

fn basin_color(smooth_iter: f32, z: vec2<f32>, max_iter: f32, n_roots: u32) -> vec3<f32> {
    if smooth_iter >= max_iter {
        return vec3<f32>(0.0, 0.0, 0.0);
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

    let shade = 1.0 / (1.0 + 0.05 * smooth_iter);

    switch params.palette {
        case 1u: {
            // Oklab basin: golden-angle hue spacing at constant lightness
            let golden = 0.618033988749895;
            let hue_angle = f32(root_id) * golden * 6.28318;
            let C = 0.13 * (0.6 + 0.4 * shade);
            let linear = oklab_to_linear_srgb(0.5 + 0.35 * shade, C * cos(hue_angle), C * sin(hue_angle));
            return pow(linear, vec3<f32>(1.0 / 2.2));
        }
        default: {
            // HSV basin (classic)
            let golden = 0.618033988749895;
            let hue = fract(f32(root_id) * golden);
            let sat = 0.55 + 0.4 * shade;
            return hsv_to_rgb(hue, sat, shade);
        }
    }
}

// -- Main ---------------------------------------------------------------------

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

    var color: vec3<f32>;
    if params.color_mode == 0u {
        color = escape_color(smooth_iter, max_iter);
    } else {
        color = basin_color(smooth_iter, z, max_iter, params.num_roots);
    }

    // Linearize before accumulation (sRGB → linear) for gamma-correct averaging.
    let linear = pow(color, vec3<f32>(2.2));
    accum[idx] = prev + vec4<f32>(linear * wt, wt);
}
