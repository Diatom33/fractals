// Escape-time fractal compute shader with emulated double precision.
// Handles: Mandelbrot (0), Julia (1), Burning Ship (2), Multibrot (3)
// Types 0-2 use double-single (f32 pair) arithmetic for deep zoom precision.
// Type 3 (Multibrot) falls back to f32 due to cpow complexity.

struct Params {
    center_hi: vec2<f32>,        // center of view (high part of double-single)
    center_lo: vec2<f32>,        // center of view (low part, sub-ULP precision)
    pixel_step: vec2<f32>,       // complex units per pixel
    resolution: vec2<u32>,       // output width, height
    max_iter: u32,
    fractal_type: u32,
    julia_c: vec2<f32>,
    power: f32,
    relaxation: f32,
    color_mode: u32,
    num_roots: u32,
    sample_offset: vec2<f32>,    // sub-pixel offset in pixel units
    sample_weight: f32,
    stride: u32,
    palette: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> iterations: array<f32>;
@group(0) @binding(2) var<storage, read_write> final_z: array<vec2<f32>>;

// ── Double-single arithmetic ─────────────────────────────────────────────────
// Each value is represented as (hi, lo) where value = hi + lo.
// This gives ~48 bits of mantissa (vs 23 for plain f32).

// Exact sum: a + b = s + e (Knuth two-sum)
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return vec2<f32>(s, e);
}

// Exact product: a * b = p + e (uses hardware FMA)
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

// Double-single add: (a.x + a.y) + (b.x + b.y)
fn ds_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    let e = a.y + b.y + s.y;
    return two_sum(s.x, e);
}

// Double-single multiply: (a.x + a.y) * (b.x + b.y)
fn ds_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = two_prod(a.x, b.x);
    let e = fma(a.x, b.y, fma(a.y, b.x, p.y));
    return two_sum(p.x, e);
}

// ── Complex arithmetic (f32) for Multibrot fallback ─────────────────────────

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn csqr(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
}

fn cmag2(z: vec2<f32>) -> f32 {
    return z.x * z.x + z.y * z.y;
}

// Complex power z^d for integer d via repeated squaring.
fn cpow_int(z_in: vec2<f32>, d: u32) -> vec2<f32> {
    var result = vec2<f32>(1.0, 0.0);
    var base = z_in;
    var exp = d;
    loop {
        if exp == 0u { break; }
        if (exp & 1u) == 1u {
            result = cmul(result, base);
        }
        base = cmul(base, base);
        exp >>= 1u;
    }
    return result;
}

// Complex power z^d for fractional d via polar form.
fn cpow(z: vec2<f32>, d: f32) -> vec2<f32> {
    let r = sqrt(cmag2(z));
    if r < 1e-20 { return vec2<f32>(0.0, 0.0); }
    let theta = atan2(z.y, z.x);
    let new_r = pow(r, d);
    let new_theta = theta * d;
    return vec2<f32>(new_r * cos(new_theta), new_r * sin(new_theta));
}

// ── Main ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.resolution.x;
    let h = params.resolution.y;
    if x >= w || y >= h { return; }

    let idx = y * params.stride + x;

    // Compute pixel coordinate in double-single precision
    // pixel = center + (pixel_index - (resolution-1)/2) * pixel_step
    let dx_pixels = f32(x) + params.sample_offset.x - f32(w - 1u) * 0.5;
    let dy_pixels = f32(y) + params.sample_offset.y - f32(h - 1u) * 0.5;
    let dx = two_prod(dx_pixels, params.pixel_step.x);
    let dy = two_prod(dy_pixels, params.pixel_step.y);
    let pixel_r = ds_add(vec2<f32>(params.center_hi.x, params.center_lo.x), dx);
    let pixel_i = ds_add(vec2<f32>(params.center_hi.y, params.center_lo.y), dy);

    // Multibrot uses f32 fallback (cpow too complex for double-single)
    if params.fractal_type == 3u {
        let pixel = vec2<f32>(pixel_r.x, pixel_i.x);
        var z = vec2<f32>(0.0, 0.0);
        let c = pixel;
        let escape_r2: f32 = 256.0;
        let max_i = params.max_iter;
        var iter: u32 = max_i;

        for (var i: u32 = 0u; i < max_i; i++) {
            let d = u32(params.power);
            if d == 2u {
                z = csqr(z) + c;
            } else {
                z = cpow_int(z, d) + c;
            }
            if cmag2(z) > escape_r2 {
                iter = i;
                break;
            }
        }

        var smooth_val: f32;
        if iter < max_i {
            let log_zn = log2(cmag2(z)) * 0.5;
            smooth_val = f32(iter) + 1.0 - log2(log_zn) / log2(params.power);
        } else {
            smooth_val = f32(max_i);
        }
        iterations[idx] = smooth_val;
        final_z[idx] = z;
        return;
    }

    // ── Double-single iteration for Mandelbrot / Julia / Burning Ship ────────

    var zr: vec2<f32>;
    var zi: vec2<f32>;
    var cr: vec2<f32>;
    var ci: vec2<f32>;

    switch params.fractal_type {
        case 1u: {
            // Julia: z₀ = pixel (double-single), c = julia_c (f32 only)
            zr = pixel_r;
            zi = pixel_i;
            cr = vec2<f32>(params.julia_c.x, 0.0);
            ci = vec2<f32>(params.julia_c.y, 0.0);
        }
        default: {
            // Mandelbrot (0), Burning Ship (2): z₀ = 0, c = pixel
            zr = vec2<f32>(0.0, 0.0);
            zi = vec2<f32>(0.0, 0.0);
            cr = pixel_r;
            ci = pixel_i;
        }
    }

    let escape_r2: f32 = 256.0;
    let max_i = params.max_iter;
    var iter: u32 = max_i;

    for (var i: u32 = 0u; i < max_i; i++) {
        // Burning Ship: take absolute value of each component before squaring
        if params.fractal_type == 2u {
            if zr.x < 0.0 { zr = vec2<f32>(-zr.x, -zr.y); }
            if zi.x < 0.0 { zi = vec2<f32>(-zi.x, -zi.y); }
        }

        // z = z² + c (double-single complex arithmetic)
        let zr2 = ds_mul(zr, zr);       // zr²
        let zi2 = ds_mul(zi, zi);       // zi²
        let zri = ds_mul(zr, zi);       // zr * zi
        let two_zri = ds_add(zri, zri); // 2 * zr * zi

        // new_zr = zr² - zi² + cr
        let diff = ds_add(zr2, vec2<f32>(-zi2.x, -zi2.y));
        zr = ds_add(diff, cr);

        // new_zi = 2*zr*zi + ci
        zi = ds_add(two_zri, ci);

        // Escape test (f32 is sufficient for |z|² > 256)
        if zr.x * zr.x + zi.x * zi.x > escape_r2 {
            iter = i;
            break;
        }
    }

    // Smooth iteration count (f32 precision is fine here)
    var smooth_val: f32;
    if iter < max_i {
        let mag2 = zr.x * zr.x + zi.x * zi.x;
        let log_zn = log2(mag2) * 0.5;
        smooth_val = f32(iter) + 1.0 - log2(log_zn);
    } else {
        smooth_val = f32(max_i);
    }

    iterations[idx] = smooth_val;
    final_z[idx] = vec2<f32>(zr.x, zi.x);
}
