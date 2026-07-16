// ── Shared palette library ────────────────────────────────────────────────
// Spliced into colorize.wgsl and median_finalize.wgsl by src/shaders.rs at
// the //INCLUDE:palettes marker. Everything here may reference the bindings
// both shaders declare with identical names: params (uniform), iterations,
// orbit_traps, roots.
//
// escape_color entry point:
//   escape_color(smooth_iter, z, dz_log2, dz_angle, px, py, sample_base)
// dz_log2 = log2(|dz/dc|) from the iterate shaders (extended-range tracking,
// valid at any iteration count and through perturbation/BLA); DZ_NONE when no
// derivative is available (Multibrot, Newton). dz_angle = arg(dz).
// sample_base is the offset into `iterations` of the sample slot to use for
// screen-space gradient palettes (colorize passes the current sample's slot,
// median passes 0 = the first sample's landscape).
//
// Palettes 11+ additionally read orbit_traps as (stripe_avg, stripe_prev,
// interior_field, 0) — see aux_mode() in the iterate shaders — and color
// interior pixels via interior_color().

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

// -- fBm noise (Storm lightning masking) ----------------------------------------

// Hash function: maps 2D point to pseudo-random value in [0,1]
fn hash2d(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise: smooth interpolation of hashed grid values
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Quintic Hermite interpolation for smoother derivatives
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let n00 = hash2d(i + vec2<f32>(0.0, 0.0));
    let n10 = hash2d(i + vec2<f32>(1.0, 0.0));
    let n01 = hash2d(i + vec2<f32>(0.0, 1.0));
    let n11 = hash2d(i + vec2<f32>(1.0, 1.0));

    let nx0 = mix(n00, n10, u.x);
    let nx1 = mix(n01, n11, u.x);
    return mix(nx0, nx1, u.y);
}

// Fractal Brownian motion with dynamic octave count
fn fbm_noise(p: vec2<f32>, num_octaves: i32) -> f32 {
    var total: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var freq: f32 = 1.0;
    var max_val: f32 = 0.0;
    for (var i: i32 = 0; i < num_octaves; i++) {
        total += amplitude * value_noise(p * freq);
        max_val += amplitude;
        amplitude *= 0.5;
        freq *= 2.0;
    }
    return total / max_val;
}

// ── Derivative-based signal helpers ───────────────────────────────────────

// Sentinel in final_z.z when the iterate shader tracked no derivative.
const DZ_NONE_THRESHOLD: f32 = -1.0e29;

fn dz_available(dz_log2: f32) -> bool {
    return dz_log2 > DZ_NONE_THRESHOLD;
}

// arg(dz) winds roughly once per iteration, so at perturbation depths (huge
// iteration counts) it decorrelates between adjacent pixels into noise.
// Palettes that use the RAW angle should fall back to arg(z) there; the
// DIFFERENCE arg(z) - arg(dz) (relief normal) stays smooth at any depth.
fn dz_angle_usable(dz_log2: f32) -> bool {
    return dz_available(dz_log2) && params.pixel_step_log2 > -23.0;
}

// log2 of the exterior distance estimate in PIXEL units.
// DE = |z|·ln|z| / |dz|; dividing by the true pixel step makes it
// zoom-invariant, so DE-driven features keep constant screen width.
fn de_log2_px(z: vec2<f32>, dz_log2: f32) -> f32 {
    let mag2 = max(dot(z, z), 1.000001);
    let log2_z = 0.5 * log2(mag2);
    let ln_z = max(log2_z * 0.69314718, 1e-9);
    return log2_z + log2(ln_z) - dz_log2 - params.pixel_step_log2;
}

// Boundary distance in pixels, clamped to [0, +inf) in log space.
fn de_px(z: vec2<f32>, dz_log2: f32) -> f32 {
    return exp2(clamp(de_log2_px(z, dz_log2), -20.0, 24.0));
}

// Relief lighting from the escape-direction normal u = z/dz (only the
// direction matters). azimuth = light direction, height = how high the sun
// sits (higher = flatter, lower = more dramatic shadows). Returns [0, 1].
fn relief_shade(z: vec2<f32>, dz_angle: f32, azimuth: f32, height: f32) -> f32 {
    let ang = atan2(z.y, z.x) - dz_angle;
    let u = vec2<f32>(cos(ang), sin(ang));
    let l = vec2<f32>(cos(azimuth), sin(azimuth));
    let t = (dot(u, l) + height) / (1.0 + height);
    return clamp(t, 0.0, 1.0);
}

// Interpolated stripe-average, computed by the iterate shaders into
// orbit_traps.xy for palettes 11+. Blending the n and n-1 averages with the
// fractional iteration kills last-iteration banding.
fn stripe_value(idx: u32, smooth_iter: f32) -> f32 {
    let aux = orbit_traps[idx];
    return mix(aux.y, aux.x, clamp(fract(smooth_iter), 0.0, 1.0));
}

// Screen-space gradient of the smooth-iteration landscape at (px, py) within
// the sample slot starting at sample_base. High magnitude ⇒ near the boundary.
fn iter_gradient(px: u32, py: u32, sample_base: u32) -> vec2<f32> {
    let stride_val = params.stride;
    var gx: f32 = 0.0;
    var gy: f32 = 0.0;
    if px > 0u && px < params.resolution.x - 1u {
        gx = iterations[sample_base + py * stride_val + px + 1u]
           - iterations[sample_base + py * stride_val + px - 1u];
    }
    if py > 0u && py < params.resolution.y - 1u {
        gy = iterations[sample_base + (py + 1u) * stride_val + px]
           - iterations[sample_base + (py - 1u) * stride_val + px];
    }
    return vec2<f32>(gx, gy);
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
fn palette_thin_film(smooth_iter: f32, z: vec2<f32>, dz_log2: f32, dz_angle: f32) -> vec3<f32> {
    let k = params.coloring_param; // angular lobe count
    let log_iter = log2(smooth_iter + 1.0);
    let t_base = sqrt(log_iter * 0.5);

    // Use derivative angle for directional bands (smooth, no iteration banding)
    // Falls back to arg(z) when unavailable (Multibrot/Newton) or too deep.
    var viewing: f32;
    if dz_angle_usable(dz_log2) {
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

// Palette 6: Storm Threshold — dark atmosphere with bright lightning at steep iteration gradients.
// fBm noise masks lightning into sparse patches. Noise lives in complex-plane coordinates
// but is reconstructed via screen_offset/cell_px + noise_seed to avoid f32 precision loss.
// noise_seed is computed on the CPU from rug::Float center, so panning is perfectly smooth.
fn palette_storm(smooth_iter: f32, px: u32, py: u32, sample_base: u32) -> vec3<f32> {
    let noise_scale = params.coloring_param;

    // Finite-difference gradient of smooth iteration count
    let grad = iter_gradient(px, py, sample_base);
    let grad_mag = length(grad);

    // Noise in complex-plane coords, reconstructed without f32 precision loss.
    // screen_offset / cell_px = pixel's fractional position in noise grid.
    // noise_seed = fract(center / cell_size) computed on CPU with arbitrary precision.
    // Together: (px - w/2)/cell_px + seed ≡ z / cell_size (mod 1 in the seed).
    // Panning shifts the seed smoothly; zooming changes cell_size proportionally.
    let cell_px = 55.0 * noise_scale;
    let noise_pos = vec2<f32>(
        (f32(px) - f32(params.resolution.x) * 0.5) / cell_px + params.noise_seed.x,
        (f32(py) - f32(params.resolution.y) * 0.5) / cell_px + params.noise_seed.y
    );

    let noise_val = fbm_noise(noise_pos, 5);
    let mask = smoothstep(0.35, 0.65, noise_val);

    // Sigmoid contrast on base value
    let log_iter = log2(smooth_iter + 1.0);
    let x_val = fract(log_iter * 0.06);
    let v = 1.0 / (1.0 + exp(-10.0 * (x_val - 0.5)));

    // Visible base: storm blue → steel grey → bronze
    let hue = 220.0 / 360.0 + 0.08 * v;
    let sat = 0.40 - 0.15 * v;
    let val = 0.12 + 0.22 * v;
    let base = hsv_to_rgb(hue, sat, val);

    // Edge glow: dim purple at steep gradients, partially masked
    let edge_glow = smoothstep(0.5, 2.0, grad_mag);
    let glow_mask = mix(0.3, 1.0, mask);
    let glow_color = vec3<f32>(0.25, 0.12, 0.40);

    // Lightning: bright blue-violet at very steep gradients, masked by fBm
    let lightning = smoothstep(1.5, 4.0, grad_mag) * mask;
    let bolt = vec3<f32>(0.82, 0.78, 1.0);

    let with_glow = mix(base, glow_color, edge_glow * glow_mask);
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

// Palette 8: Bioluminescence — deep-sea abyssal glow with depth-aware scattering
// Emitters are structural features (high iteration gradient). Light scatters through
// dark water with wavelength-dependent kernels (green tight, blue wide, red absorbed).
// Beer-Lambert attenuation makes deep structures bluer and dimmer.
fn palette_biolum(smooth_iter: f32, px: u32, py: u32, sample_base: u32) -> vec3<f32> {
    let murkiness = params.coloring_param;
    let stride_val = params.stride;
    let h = params.resolution.y;
    let w = params.resolution.x;
    let max_iter_f = f32(params.max_iter);
    let n = smooth_iter;

    // This pixel's gradient magnitude (direct emitter signal)
    let self_emitter = length(iter_gradient(px, py, sample_base));

    // Accumulate glow from neighborhood with depth-aware, wavelength-dependent scattering
    let sigma_depth = max(murkiness, 0.5);
    var glow_r: f32 = 0.0;
    var glow_g: f32 = 0.0;
    var glow_b: f32 = 0.0;

    for (var dy: i32 = -4; dy <= 4; dy++) {
        for (var dx: i32 = -4; dx <= 4; dx++) {
            let dist_sq = f32(dx * dx + dy * dy);
            if dist_sq > 20.25 { continue; }  // circular radius 4.5

            let nx = i32(px) + dx;
            let ny = i32(py) + dy;
            if nx < 1 || nx >= i32(w) - 1 || ny < 1 || ny >= i32(h) - 1 { continue; }

            let n_neighbor = iterations[sample_base + u32(ny) * stride_val + u32(nx)];
            if n_neighbor >= max_iter_f { continue; }

            let ngx = iterations[sample_base + u32(ny) * stride_val + u32(nx + 1)]
                    - iterations[sample_base + u32(ny) * stride_val + u32(nx - 1)];
            let ngy = iterations[sample_base + (u32(ny) + 1u) * stride_val + u32(nx)]
                    - iterations[sample_base + (u32(ny) - 1u) * stride_val + u32(nx)];
            let n_emit = sqrt(ngx * ngx + ngy * ngy);

            if n_emit < 0.1 { continue; }

            let dist_val = sqrt(dist_sq);
            let depth_diff = abs(n - n_neighbor);
            let depth_w = exp(-depth_diff / sigma_depth);

            // Wavelength-dependent spatial falloff (exponential tails — murky water)
            // Red: very tight (absorbed fast), Green: moderate, Blue: wide (scatters most)
            glow_r += exp(-dist_val / 1.2) * depth_w * n_emit;
            glow_g += exp(-dist_val / 2.0) * depth_w * n_emit;
            glow_b += exp(-dist_val / 3.5) * depth_w * n_emit;
        }
    }

    let glow_norm = 0.008;

    // Direct emission: hue shifts with iteration depth — different "species" at different depths.
    // Green dominates at moderate depth, blue-violet at deep, cyan-teal at shallow.
    let log_iter = log2(n + 1.0);
    let emit_intensity = smoothstep(0.2, 2.5, self_emitter);
    let species_phase = log_iter * 0.4;
    let emit_color = vec3<f32>(
        0.04 + 0.08 * max(sin(species_phase + 3.5), 0.0),  // faint warm at some depths
        0.3 + 0.55 * (0.5 + 0.5 * cos(species_phase)),      // green peaks periodically
        0.3 + 0.45 * (0.5 + 0.5 * sin(species_phase))       // blue peaks offset from green
    );

    let direct = emit_color * emit_intensity;

    // Scattered glow (wavelength-shifted: green center, blue edges)
    let scattered = vec3<f32>(
        glow_r * glow_norm * 0.15,
        glow_g * glow_norm * 0.7,
        glow_b * glow_norm * 0.55
    );

    // Beer-Lambert depth attenuation (wavelength-dependent)
    let depth_atten = log_iter * 0.12;
    let atten = vec3<f32>(
        exp(-depth_atten * murkiness * 0.4),
        exp(-depth_atten * murkiness * 0.12),
        exp(-depth_atten * murkiness * 0.08)
    );

    // Dark abyssal water + faint ambient marine snow
    let water = vec3<f32>(0.003, 0.006, 0.018);
    let ambient = vec3<f32>(0.0, 0.008, 0.015) * (0.3 + 0.2 * sin(log_iter * 0.7));

    return clamp(water + ambient + (direct + scattered) * atten, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Palette 9: STEVE — the Mandelbrot set boundary IS the STEVE ribbon (mauve glow
// like Bioluminescence's boundary detection); the exterior is the picket-fence
// field, with each green picket oriented parallel to the local escape direction
// (like thin-film's dz_angle bands).
fn palette_steve(smooth_iter: f32, z: vec2<f32>, dz_log2: f32, dz_angle: f32, px: u32, py: u32, sample_base: u32) -> vec3<f32> {
    // "Activity" = atmospheric intensity (analogous to Biolum's murkiness).
    // Scales ribbon width, halo spread, and fence brightness together so the
    // aurora can go from faint post-glimmer to bright charged-sky vibe.
    let activity = params.coloring_param;
    let k = 18.0;                                    // post density (fixed)

    // ── Boundary detection via ∇(smooth_iter). High gradient ⇒ near set boundary.
    let grad_mag = length(iter_gradient(px, py, sample_base));

    // ── STEVE ribbon: mauve band on the boundary, with "species"-style hue
    //    drift by iter depth (like Biolum does). The ribbon shifts between
    //    deep-plum → steve-mauve → cool blue-violet → warm pink-mauve as the
    //    boundary winds through regions of different local period. A second
    //    slower cycle adds a lightness pulse so peaks glow more hot-white.
    // Ribbon width scales with activity: calm aurora → thin boundary line,
    // charged aurora → wide glowing band.
    let ribbon = smoothstep(0.6 / activity, 4.0 / activity, grad_mag);
    let log_iter = log2(smooth_iter + 1.0);
    let phase_hue = log_iter * 0.35;
    let phase_lum = log_iter * 0.17;
    let mauve_plum   = vec3<f32>(0.290, 0.122, 0.322); // #4A1F52
    let mauve_steve  = vec3<f32>(0.706, 0.541, 0.831); // #B48AD4
    let mauve_cool   = vec3<f32>(0.537, 0.561, 0.878); // cool violet-blue
    let mauve_warm   = vec3<f32>(0.895, 0.663, 0.828); // warm pink-mauve
    let mauve_white  = vec3<f32>(0.957, 0.863, 0.973); // #F4DCF8
    // 3-way hue drift: cool ↔ steve-mauve ↔ warm, driven by sin & cos.
    let w_cool = max(cos(phase_hue), 0.0);
    let w_warm = max(cos(phase_hue + 3.14159265), 0.0);
    let w_mid  = max(sin(phase_hue) * sin(phase_hue), 0.15);
    let w_sum  = w_cool + w_warm + w_mid;
    let ribbon_mid = (mauve_cool * w_cool + mauve_warm * w_warm + mauve_steve * w_mid) / w_sum;
    // Luminance pulse: occasionally flare toward #F4DCF8 (near-white mauve).
    let lum_pulse = 0.5 + 0.5 * cos(phase_lum);
    let ribbon_inner = mix(ribbon_mid, mauve_white, 0.28 * lum_pulse);
    // Deep plum at the outer edge of the ribbon (low grad), bright mauve at core.
    let ribbon_col = mix(mauve_plum, ribbon_inner, ribbon);

    // ── Picket fence (exterior): posts parallel to the escape direction.
    //    dz_angle is arg(dz/dc), the direction the orbit escapes through.
    //    Each post's phase combines angle + log_iter so posts have finite
    //    vertical extent (in iter axis) instead of one long radial ridge.
    var escape_angle: f32;
    if dz_angle_usable(dz_log2) {
        escape_angle = dz_angle;
    } else {
        escape_angle = atan2(z.y, z.x);
    }
    let phase = k * 0.5 * escape_angle + log_iter * 2.8;
    // Sharpened peak: pow of shifted cosine gives narrow bright posts with
    // proper dark sky between them (not pink/green alternating taffy).
    let peak = 0.5 + 0.5 * cos(phase);
    let core = pow(peak, 18.0);          // very narrow bright core
    // Halo spread widens with activity (soft glow around each post).
    let halo_exp = max(1.5, 6.0 / activity);
    let halo = pow(peak, halo_exp) * 0.35 * activity;
    // Clamp so high activity doesn't blow posts to pure white — the blend
    // below uses this as a weight and [0, 1] keeps colors in gamut.
    let fence_intensity = min(core + halo, 1.0);

    // Post color: body shifts along the post's height (driven by log_iter)
    // from teal-green low → bright green mid → pink flush at tip. This is
    // what gives pickets a proper vertical gradient instead of flat bands.
    let hue_cycle = params.coloring_param_2;  // post color cycle rate
    let tip_phase = fract(log_iter * hue_cycle);
    let green_teal  = vec3<f32>(0.110, 0.780, 0.620);   // deep teal-green
    let green_body  = vec3<f32>(0.235, 0.910, 0.533);   // #3CE888
    let green_light = vec3<f32>(0.612, 1.000, 0.722);   // #9CFFB8
    let pink_flush  = vec3<f32>(0.941, 0.659, 0.784);   // #F0A8C8
    var post_col: vec3<f32>;
    if tip_phase < 0.45 {
        post_col = mix(green_teal, green_body, smoothstep(0.0, 0.45, tip_phase));
    } else if tip_phase < 0.80 {
        post_col = mix(green_body, green_light, smoothstep(0.45, 0.80, tip_phase));
    } else {
        post_col = mix(green_light, pink_flush, smoothstep(0.80, 1.00, tip_phase));
    }
    // Dark sky between posts (near-black violet).
    let sky = vec3<f32>(0.024, 0.008, 0.071);
    let exterior = sky + post_col * fence_intensity * (1.0 - ribbon);
    return mix(exterior, ribbon_col, ribbon);
}

// Palette 10: Inverted Pair — high-contrast sinusoidal bands between complementary colors.
// Fast axis: sinusoidal oscillation between dark A and bright B.
// Slow axis: (A, B) drifts through complementary hue pairs (180° apart in Oklab a/b plane).
// At H_slow=0, chroma ≈ 0 → pure black/white; grows as hue rotates for subtly tinted pairs.
fn palette_inverted_pair(smooth_iter: f32) -> vec3<f32> {
    let pi = 3.14159265;
    let fast_freq = params.coloring_param;
    let slow_freq = 0.0025;

    let h_slow = fract(smooth_iter * slow_freq);
    // Chroma grows from 0 (pure B&W) as hue rotates; sin(π·h) peaks mid-cycle, returns to 0.
    let chroma = 0.15 * sin(pi * h_slow);

    let hue_angle = 2.0 * pi * h_slow;
    let l_low = 0.10;
    let l_high = 0.92;

    // A = dark with hue H_slow; B = bright with complementary hue (H_slow + 0.5).
    let a_a = chroma * cos(hue_angle);
    let a_b = chroma * sin(hue_angle);
    let b_a = -a_a;
    let b_b = -a_b;

    let t = 0.5 + 0.5 * sin(smooth_iter * 2.0 * pi * fast_freq);

    let L = mix(l_low, l_high, t);
    let ok_a = mix(a_a, b_a, t);
    let ok_b = mix(a_b, b_b, t);

    let linear = oklab_to_linear_srgb(L, ok_a, ok_b);
    return linear_to_srgb(linear);
}

// Palette 11: Obsidian — relief-lit volcanic glass. The exterior is a dark
// cooled lava field embossed by the escape-direction normal, with a tight
// warm specular glint; the set boundary smolders through as an ember rim
// whose screen width is zoom-invariant (distance estimation).
fn palette_obsidian(smooth_iter: f32, z: vec2<f32>, dz_log2: f32, dz_angle: f32) -> vec3<f32> {
    let azimuth = params.coloring_param;
    let log_iter = log2(smooth_iter + 1.0);
    // cold green-grey glass, slowly breathing with depth
    let base = vec3<f32>(0.020, 0.028, 0.030) + vec3<f32>(0.008, 0.012, 0.010) * cos(log_iter * 0.4);
    var col = base;
    if dz_available(dz_log2) {
        let shade = relief_shade(z, dz_angle, azimuth, 1.2);
        // diffuse sheen
        col = base + vec3<f32>(0.10, 0.13, 0.15) * shade * shade;
        // tight torchlight glint on the glass
        let spec = pow(shade, 48.0);
        col += vec3<f32>(0.90, 0.82, 0.70) * spec * 0.8;
        // ember rim: the boundary glows through cracks in the crust
        let de = de_px(z, dz_log2);
        let ember_w = max(params.coloring_param_2, 0.25);
        let ember = exp(-de / ember_w);
        col += vec3<f32>(1.00, 0.30, 0.05) * ember * ember * 1.2;
        // heat haze further from the crack
        col += vec3<f32>(0.30, 0.05, 0.02) * exp(-de / (ember_w * 4.0)) * 0.35;
    }
    return clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Palette 12: Noctilucent — night-shining clouds. Stripe-average wisps in
// electric silver-blue drift over a deep twilight sky, with a faint cold
// sheen where the set boundary approaches.
fn palette_noctilucent(smooth_iter: f32, z: vec2<f32>, dz_log2: f32, idx: u32) -> vec3<f32> {
    let contrast = max(params.coloring_param_2, 0.1);
    let s = clamp(stripe_value(idx, smooth_iter), 0.0, 1.0);
    let log_iter = log2(smooth_iter + 1.0);
    // twilight gradient: deep blue breathing toward horizon purple
    let sky = mix(
        vec3<f32>(0.012, 0.020, 0.055),
        vec3<f32>(0.045, 0.020, 0.075),
        0.5 + 0.5 * cos(log_iter * 0.25),
    );
    // wisp field, contrast-shaped
    let wisp = pow(s, contrast * 2.0);
    // silver at high altitude (shallow), electric blue deeper in
    let cloud = mix(
        vec3<f32>(0.55, 0.75, 1.00),
        vec3<f32>(0.85, 0.93, 1.00),
        0.5 + 0.5 * cos(log_iter * 0.18 + 1.0),
    );
    var col = sky + cloud * wisp * 0.9;
    if dz_available(dz_log2) {
        let de = de_px(z, dz_log2);
        col += vec3<f32>(0.35, 0.55, 0.90) * exp(-de / 3.0) * 0.5;
    }
    return clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Palette 13: Lichtenberg — fossilized lightning. Thin amber filaments branch
// where the stripe field crosses its midline; the set boundary itself is the
// white-hot trunk the discharge grew from.
fn palette_lichtenberg(smooth_iter: f32, z: vec2<f32>, dz_log2: f32, idx: u32) -> vec3<f32> {
    let sharp = max(params.coloring_param_2, 1.0);
    let s = clamp(stripe_value(idx, smooth_iter), 0.0, 1.0);
    // filaments at the stripe midline crossings
    let line = pow(1.0 - abs(2.0 * s - 1.0), sharp);
    let log_iter = log2(smooth_iter + 1.0);
    // charred substrate
    let substrate = vec3<f32>(0.030, 0.016, 0.010) + vec3<f32>(0.020, 0.010, 0.004) * cos(log_iter * 0.3);
    let amber = vec3<f32>(1.00, 0.55, 0.12);
    let core  = vec3<f32>(1.00, 0.90, 0.60);
    var col = substrate + amber * line * 0.85 + core * pow(line, 4.0) * 0.7;
    if dz_available(dz_log2) {
        let de = de_px(z, dz_log2);
        let trunk = exp(-de / 2.0);
        col += vec3<f32>(1.00, 0.45, 0.08) * trunk * 0.9;
        col += vec3<f32>(1.00, 0.85, 0.50) * trunk * trunk * 0.6;
    }
    return clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Palette 14: Corona — the set boundary as a neon plasma filament of
// constant screen width at any zoom depth (pure distance estimation),
// hue drifting slowly along the discharge, over near-black void.
fn palette_corona(smooth_iter: f32, z: vec2<f32>, dz_log2: f32) -> vec3<f32> {
    let width = max(params.coloring_param, 0.25);   // filament width in pixels
    let drift = params.coloring_param_2;
    let log_iter = log2(smooth_iter + 1.0);
    var col = vec3<f32>(0.004, 0.003, 0.010);
    if dz_available(dz_log2) {
        let de = de_px(z, dz_log2);
        let hue_angle = log_iter * drift * 6.28318;
        let neon = vec3<f32>(
            0.45 + 0.35 * cos(hue_angle + 1.2),
            0.30 + 0.30 * cos(hue_angle + 3.6),
            0.85 + 0.15 * cos(hue_angle + 5.2),
        );
        let filament = exp(-de / width);
        let halo = exp(-de / (width * 6.0));
        col += neon * halo * 0.35;
        col += neon * filament * 0.9;
        col += vec3<f32>(1.0) * pow(filament, 3.0) * 0.8;   // white-hot core
    }
    return clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Palette 15: Slot Canyon — sandstone strata carved by the stripe field and
// lit by a low warm keylight (relief shading), with distance-estimation
// crevice occlusion and reflected glow in the deepest cracks.
fn palette_slot_canyon(smooth_iter: f32, z: vec2<f32>, dz_log2: f32, dz_angle: f32, idx: u32) -> vec3<f32> {
    let azimuth = params.coloring_param;
    let s = clamp(stripe_value(idx, smooth_iter), 0.0, 1.0);
    let log_iter = log2(smooth_iter + 1.0);
    // strata: deep umber → glowing coral → pale peach
    let umber = vec3<f32>(0.26, 0.10, 0.05);
    let coral = vec3<f32>(0.85, 0.35, 0.12);
    let peach = vec3<f32>(1.00, 0.72, 0.45);
    var albedo = mix(umber, coral, smoothstep(0.15, 0.60, s));
    albedo = mix(albedo, peach, smoothstep(0.70, 0.95, s));
    // slow bounce-light drift with depth
    albedo *= 0.85 + 0.15 * cos(log_iter * 0.2);
    var col = albedo * 0.25;
    if dz_available(dz_log2) {
        let shade = relief_shade(z, dz_angle, azimuth, 0.8);
        col = albedo * (0.20 + 0.80 * shade * shade);
        // sun-kissed rim highlights
        col += vec3<f32>(1.00, 0.80, 0.55) * pow(shade, 24.0) * 0.35;
        // crevice occlusion near the boundary...
        let de = de_px(z, dz_log2);
        col *= 1.0 - 0.65 * exp(-de / 3.0);
        // ...with reflected ember glow in the deepest cracks
        col += vec3<f32>(0.55, 0.16, 0.04) * exp(-de / 1.2) * 0.6;
    }
    return clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Interior coloring (palettes 11+) ────────────────────────────────────────
// Classic palettes keep pure-black interiors. The new palettes read the
// interior field — the orbit-average of exp(-|z|), a smooth bounded value
// the iterate shaders store in orbit_traps.z — and give the set body a
// material of its own.

fn palette_has_interior(p: u32) -> bool {
    return p >= 11u;
}

fn interior_color(z: vec2<f32>, idx: u32) -> vec3<f32> {
    let f = clamp(orbit_traps[idx].z, 0.0, 1.0);
    switch params.palette {
        case 11u: {
            // polished black glass with a faint cold flow-banding
            let v = smoothstep(0.1, 0.9, f);
            return vec3<f32>(0.010, 0.012, 0.016) + vec3<f32>(0.05, 0.07, 0.09) * v * v;
        }
        case 12u: {
            // starless night sky below the clouds
            return vec3<f32>(0.004, 0.006, 0.016) + vec3<f32>(0.006, 0.010, 0.024) * f;
        }
        case 13u: {
            // charred wood
            return vec3<f32>(0.014, 0.007, 0.004) + vec3<f32>(0.020, 0.008, 0.002) * f;
        }
        case 14u: {
            // dim violet core light
            return vec3<f32>(0.012, 0.006, 0.030) + vec3<f32>(0.05, 0.02, 0.10) * smoothstep(0.2, 0.9, f);
        }
        case 15u: {
            // deep canyon shadow, faintly warm
            return vec3<f32>(0.050, 0.020, 0.012) * (0.4 + 0.6 * f);
        }
        default: { return vec3<f32>(0.0); }
    }
}

// Dispatch to selected palette (returns sRGB)
fn escape_color(smooth_iter: f32, z: vec2<f32>, dz_log2: f32, dz_angle: f32, px: u32, py: u32, sample_base: u32) -> vec3<f32> {
    switch params.palette {
        case 1u: { return palette_oklab(smooth_iter); }
        case 2u: { return palette_smooth(smooth_iter); }
        case 3u: { return palette_mono(smooth_iter); }
        case 4u: { return palette_thin_film(smooth_iter, z, dz_log2, dz_angle); }
        case 5u: { return palette_aurora(smooth_iter); }
        case 6u: { return palette_storm(smooth_iter, px, py, sample_base); }
        case 7u: { return palette_canopy(smooth_iter, py * params.stride + px); }
        case 8u: { return palette_biolum(smooth_iter, px, py, sample_base); }
        case 9u: { return palette_steve(smooth_iter, z, dz_log2, dz_angle, px, py, sample_base); }
        case 10u: { return palette_inverted_pair(smooth_iter); }
        case 11u: { return palette_obsidian(smooth_iter, z, dz_log2, dz_angle); }
        case 12u: { return palette_noctilucent(smooth_iter, z, dz_log2, py * params.stride + px); }
        case 13u: { return palette_lichtenberg(smooth_iter, z, dz_log2, py * params.stride + px); }
        case 14u: { return palette_corona(smooth_iter, z, dz_log2); }
        case 15u: { return palette_slot_canyon(smooth_iter, z, dz_log2, dz_angle, py * params.stride + px); }
        default: { return palette_classic(smooth_iter); }
    }
}

// -- Root-basin coloring (Newton/Nova) ----------------------------------------

fn basin_color(smooth_iter: f32, z: vec2<f32>, n_roots: u32) -> vec3<f32> {
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
