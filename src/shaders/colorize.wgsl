// Colorization compute shader.
// Reads iteration data + final z, produces weighted color into accumulation buffer.
// Supports both escape-time and root-basin coloring with multiple palettes.
// Called once per sub-pixel sample; accumulates into a vec4<f32> buffer (rgb + weight).
//
// Two-population compositing: interior (max_iter) and exterior samples are tracked
// separately. accum.rgb holds weighted exterior color (in Oklab L,a,b), accum.w holds
// exterior weight only. Interior samples contribute nothing to accum — their coverage
// is inferred in finalize as (total_weight - exterior_weight) / total_weight.

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
    palette: u32,            // 0=Classic, 1=Oklab, 2=Smooth, 3=Mono, 4=ThinFilm, 5=Aurora, 6=Storm
    sample_index: u32,
    num_samples: u32,
    coloring_param: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> iterations: array<f32>;
@group(0) @binding(2) var<storage, read> final_z: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> roots: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> orbit_traps: array<vec4<f32>>;

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

// -- sRGB <-> Oklab -----------------------------------------------------------

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

fn linear_srgb_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * c.x + 0.5363325363 * c.y + 0.0514459929 * c.z;
    let m = 0.2119034982 * c.x + 0.6806995451 * c.y + 0.1073969566 * c.z;
    let s = 0.0883024619 * c.x + 0.2817188376 * c.y + 0.6299787005 * c.z;

    let l_ = pow(max(l, 0.0), 1.0 / 3.0);
    let m_ = pow(max(m, 0.0), 1.0 / 3.0);
    let s_ = pow(max(s, 0.0), 1.0 / 3.0);

    let L  =  0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let a  =  1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let b  =  0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    return vec3<f32>(L, a, b);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

// Convert sRGB color to Oklab for averaging
fn srgb_to_oklab(srgb: vec3<f32>) -> vec3<f32> {
    return linear_srgb_to_oklab(srgb_to_linear(srgb));
}

// -- Escape-time palettes (return sRGB) ---------------------------------------

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
    let hue_angle = log_iter * 0.9 + 0.5;
    let L = 0.75;
    let C = 0.12;
    let a_ok = C * cos(hue_angle);
    let b_ok = C * sin(hue_angle);
    let linear = oklab_to_linear_srgb(L, a_ok, b_ok);
    return linear_to_srgb(linear);
}

// Palette 2: Smooth iq-style cosine gradient
fn palette_smooth(smooth_iter: f32) -> vec3<f32> {
    let t = log2(smooth_iter + 1.0) * 0.1;
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.00, 0.10, 0.20);
    return a + b * cos(6.28318 * (c * t + d));
}

// Palette 3: Monochrome — single cool hue, varying brightness
fn palette_mono(smooth_iter: f32) -> vec3<f32> {
    let log_iter = log2(smooth_iter + 1.0);
    let val = 0.5 + 0.5 * cos(log_iter * 0.4 + 1.0);
    return vec3<f32>(val * 0.15, val * 0.3, val);
}

// Palette 4: Thin-Film Interference (soap bubble / oil slick)
// Maps smooth_iter to "optical thickness", modulates by derivative angle for band-free directional iridescence.
// arg(dz) is smooth across iteration boundaries (unlike arg(z) which doubles each iteration).
fn palette_thin_film(smooth_iter: f32, z: vec2<f32>, dz_mag: f32, dz_angle: f32) -> vec3<f32> {
    let k = params.coloring_param; // angular lobe count
    let log_iter = log2(smooth_iter + 1.0);
    let t_base = sqrt(log_iter * 0.5);

    // Use derivative angle for directional bands (smooth, no iteration banding)
    // Falls back to arg(z) when derivative not available (perturbation/Newton)
    var viewing: f32;
    if dz_mag > 0.0 {
        viewing = abs(cos(dz_angle * k));
    } else {
        let angle = atan2(z.y, z.x);
        viewing = abs(cos(angle * k));
    }
    let t_eff = t_base / max(viewing, 0.04);

    let pi = 3.14159265;
    let r = pow(sin(pi * t_eff / 0.650), 2.0);
    let g = pow(sin(pi * t_eff / 0.550), 2.0);
    let b = pow(sin(pi * t_eff / 0.450), 2.0);
    return vec3<f32>(r, g, b);
}

// Palette 5: Midnight Aurora — narrow luminous green-violet bands on dark background
// Uses smooth cosine-based color cycling to avoid discontinuities.
fn palette_aurora(smooth_iter: f32) -> vec3<f32> {
    let freq = params.coloring_param; // band frequency
    let log_iter = log2(smooth_iter + 1.0);

    // Create soft-edged luminous bands with varying width
    let band = fract(log_iter * freq * 0.08);
    let glow = smoothstep(0.3, 0.48, band) * (1.0 - smoothstep(0.52, 0.7, band));

    // Secondary dimmer glow between main bands
    let band2 = fract(log_iter * freq * 0.08 + 0.5);
    let glow2 = smoothstep(0.35, 0.48, band2) * (1.0 - smoothstep(0.52, 0.65, band2)) * 0.3;

    // Smooth cosine-based color cycling — no if/else discontinuities
    let hue_angle = log_iter * 0.44; // ~0.07 * 2π
    let base_color = vec3<f32>(
        0.22 + 0.25 * cos(hue_angle - 4.0),   // red channel: peaks at pink/violet
        0.45 + 0.45 * cos(hue_angle),           // green: peaks at green/teal
        0.42 + 0.40 * cos(hue_angle - 2.5)     // blue: peaks at teal/violet
    );

    // Secondary hue (shifted phase)
    let hue_angle2 = log_iter * 0.44 + 2.5;
    let sec_color = vec3<f32>(
        0.15 + 0.15 * cos(hue_angle2 - 4.0),
        0.35 + 0.35 * cos(hue_angle2),
        0.35 + 0.30 * cos(hue_angle2 - 2.5)
    );

    // Dark midnight base — slightly blue-tinted
    let dark = vec3<f32>(0.008, 0.006, 0.02);
    let primary = mix(dark, base_color, glow);
    return primary + sec_color * glow2;
}

// Palette 6: Storm Threshold — dark atmosphere with bright lightning at steep iteration gradients
fn palette_storm(smooth_iter: f32, idx: u32) -> vec3<f32> {
    let steepness = params.coloring_param; // sigmoid steepness
    let stride_val = params.stride;
    let h = params.resolution.y;
    let px = idx % stride_val;
    let py = idx / stride_val;
    let base_off = params.sample_index * stride_val * h;

    // Finite-difference gradient of smooth iteration count
    var grad_x: f32 = 0.0;
    var grad_y: f32 = 0.0;
    if px > 0u && px < params.resolution.x - 1u {
        grad_x = iterations[base_off + py * stride_val + px + 1u]
               - iterations[base_off + py * stride_val + px - 1u];
    }
    if py > 0u && py < h - 1u {
        grad_y = iterations[base_off + (py + 1u) * stride_val + px]
               - iterations[base_off + (py - 1u) * stride_val + px];
    }
    let grad_mag = sqrt(grad_x * grad_x + grad_y * grad_y);

    // Sigmoid contrast on base value
    let log_iter = log2(smooth_iter + 1.0);
    let x_val = fract(log_iter * 0.06);
    let v = 1.0 / (1.0 + exp(-steepness * (x_val - 0.5)));

    // Visible base: storm blue → steel grey → bronze
    let hue = 220.0 / 360.0 + 0.08 * v;
    let sat = 0.40 - 0.15 * v;
    let val = 0.12 + 0.22 * v;
    let base = hsv_to_rgb(hue, sat, val);

    // Edge glow: dim purple at genuinely steep gradients
    let edge_glow = smoothstep(0.5, 2.0, grad_mag);
    let glow_color = vec3<f32>(0.25, 0.12, 0.40);

    // Lightning: bright blue-white only at very steep gradients
    let lightning = smoothstep(1.5, 4.0, grad_mag);
    let bolt = vec3<f32>(0.90, 0.90, 1.0);

    let with_glow = mix(base, glow_color, edge_glow);
    return mix(with_glow, bolt, lightning);
}

// Palette 7: Canopy — bright bokeh highlights from orbit traps, max-channel normalized
fn palette_canopy(smooth_iter: f32, idx: u32) -> vec3<f32> {
    let traps = orbit_traps[idx];
    let log_iter = log2(smooth_iter + 1.0);

    let canopy_phase = log_iter * 0.15;
    let canopy = vec3<f32>(
        0.12 + 0.08 * cos(canopy_phase + 0.5),
        0.15 + 0.10 * cos(canopy_phase),
        0.04 + 0.03 * cos(canopy_phase + 1.0)
    );

    let ruby     = vec3<f32>(0.85, 0.12, 0.15);
    let sapphire = vec3<f32>(0.15, 0.20, 0.90);
    let amber    = vec3<f32>(0.90, 0.65, 0.10);
    let emerald  = vec3<f32>(0.10, 0.80, 0.30);

    let trap_scale = params.coloring_param;
    let i0 = exp(-traps.x * trap_scale);
    let i1 = exp(-traps.y * trap_scale);
    let i2 = exp(-traps.z * trap_scale);
    let i3 = exp(-traps.w * trap_scale);

    let max_i = max(max(i0, i1), max(i2, i3));
    let canopy_brightness = 0.6 + 0.4 * cos(log_iter * 0.08);

    // Raw weighted jewel sum — allows bright highlights (bokeh effect)
    let raw = ruby * i0 + sapphire * i1 + amber * i2 + emerald * i3;
    // Only normalize when sum exceeds displayable range — preserves full brightness
    let max_ch = max(raw.x, max(raw.y, raw.z));
    let jewels = select(raw, raw / max_ch, max_ch > 1.0);

    return mix(canopy * canopy_brightness, jewels, max_i);
}

// Dispatch to selected palette (returns sRGB)
fn escape_color(smooth_iter: f32, max_iter: f32, z: vec2<f32>, dz_mag: f32, dz_angle: f32, idx: u32) -> vec3<f32> {
    switch params.palette {
        case 1u: { return palette_oklab(smooth_iter); }
        case 2u: { return palette_smooth(smooth_iter); }
        case 3u: { return palette_mono(smooth_iter); }
        case 4u: { return palette_thin_film(smooth_iter, z, dz_mag, dz_angle); }
        case 5u: { return palette_aurora(smooth_iter); }
        case 6u: { return palette_storm(smooth_iter, idx); }
        case 7u: { return palette_canopy(smooth_iter, idx); }
        default: { return palette_classic(smooth_iter); }
    }
}

// -- Root-basin coloring (Newton/Nova) ----------------------------------------

fn basin_color(smooth_iter: f32, z: vec2<f32>, max_iter: f32, n_roots: u32) -> vec3<f32> {
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
            let golden = 0.618033988749895;
            let hue_angle = f32(root_id) * golden * 6.28318;
            let C = 0.13 * (0.6 + 0.4 * shade);
            let linear = oklab_to_linear_srgb(0.5 + 0.35 * shade, C * cos(hue_angle), C * sin(hue_angle));
            return linear_to_srgb(linear);
        }
        default: {
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
    let iter_idx = params.sample_index * params.stride * h + idx;
    let smooth_iter = iterations[iter_idx];
    let fz = final_z[idx];
    let z = fz.xy;
    let dz_mag = fz.z;
    let dz_angle = fz.w;
    let wt = params.sample_weight;
    let prev = accum[idx];

    // Interior samples (hit max_iter): don't accumulate any color.
    // Their coverage is inferred in finalize as (total_weight - exterior_weight) / total_weight.
    let is_interior = smooth_iter >= max_iter;
    if is_interior {
        // Interior contributes nothing to accum — coverage tracked implicitly
        return;
    }

    // Exterior sample: compute color, convert to Oklab for perceptually uniform averaging
    var srgb_color: vec3<f32>;
    if params.color_mode == 0u {
        srgb_color = escape_color(smooth_iter, max_iter, z, dz_mag, dz_angle, idx);
    } else {
        srgb_color = basin_color(smooth_iter, z, max_iter, params.num_roots);
    }

    // Average in Oklab space for perceptual uniformity
    let oklab = srgb_to_oklab(srgb_color);

    // Accumulate: rgb = weighted Oklab (L,a,b), w = exterior weight only
    accum[idx] = prev + vec4<f32>(oklab * wt, wt);
}
