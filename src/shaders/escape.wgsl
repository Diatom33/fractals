// Escape-time fractal compute shader with emulated double precision.
// Handles: Mandelbrot (0), Julia (1), Burning Ship (2), Multibrot (3),
//          Tricorn (7), Celtic (8), Perpendicular (9), Buffalo (10)
// All types use double-single (f32 pair) arithmetic for deep zoom precision.
// Type 3 (Multibrot) uses ds_cpow_int for integer powers via repeated squaring.

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
    sample_index: u32,
    num_samples: u32,
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

// ── Double-single complex power z^n for integer n (repeated squaring) ───────

fn ds_cpow_int(zr_in: vec2<f32>, zi_in: vec2<f32>, n: u32) -> array<vec2<f32>, 2> {
    var rr = vec2<f32>(1.0, 0.0);  // result real = 1
    var ri = vec2<f32>(0.0, 0.0);  // result imag = 0
    var br = zr_in;                 // base real
    var bi = zi_in;                 // base imag
    var exp = n;
    loop {
        if exp == 0u { break; }
        if (exp & 1u) == 1u {
            // result = result * base (complex multiply in double-single)
            let rr_br = ds_mul(rr, br);
            let ri_bi = ds_mul(ri, bi);
            let rr_bi = ds_mul(rr, bi);
            let ri_br = ds_mul(ri, br);
            let new_rr = ds_add(rr_br, vec2<f32>(-ri_bi.x, -ri_bi.y));
            let new_ri = ds_add(rr_bi, ri_br);
            rr = new_rr;
            ri = new_ri;
        }
        // base = base * base (complex squaring in double-single)
        let br_br = ds_mul(br, br);
        let bi_bi = ds_mul(bi, bi);
        let br_bi = ds_mul(br, bi);
        let new_br = ds_add(br_br, vec2<f32>(-bi_bi.x, -bi_bi.y));
        let new_bi = ds_add(br_bi, br_bi);  // 2*br*bi
        br = new_br;
        bi = new_bi;
        exp >>= 1u;
    }
    return array<vec2<f32>, 2>(rr, ri);
}

// ── Complex arithmetic (f32) for fractional power fallback ──────────────────

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

    // ── Double-single iteration for Mandelbrot / Julia / Burning Ship / Multibrot ──

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
            // Mandelbrot (0), Burning Ship (2), Multibrot (3), Tricorn (7),
            // Celtic (8), Perpendicular (9), Buffalo (10): z₀ = 0, c = pixel
            zr = vec2<f32>(0.0, 0.0);
            zi = vec2<f32>(0.0, 0.0);
            cr = pixel_r;
            ci = pixel_i;
        }
    }

    let escape_r2: f32 = 256.0;
    let max_i = params.max_iter;
    var iter: u32 = max_i;

    // Multibrot power (only used when fractal_type == 3)
    let multibrot_d = u32(params.power);

    for (var i: u32 = 0u; i < max_i; i++) {

        // ── Multibrot (type 3): z = z^d + c using ds_cpow_int ───────────
        if params.fractal_type == 3u {
            let zd = ds_cpow_int(zr, zi, multibrot_d);
            zr = ds_add(zd[0], cr);
            zi = ds_add(zd[1], ci);
        } else {
            // ── z² variants: Mandelbrot, Julia, Burning Ship, etc. ──────

            // Burning Ship: take absolute value of each component before squaring
            if params.fractal_type == 2u {
                if zr.x < 0.0 { zr = vec2<f32>(-zr.x, -zr.y); }
                if zi.x < 0.0 { zi = vec2<f32>(-zi.x, -zi.y); }
            }

            // Save sign of zr before any modifications (needed for Perpendicular)
            let zr_sign = zr.x;

            // z = z² + c (double-single complex arithmetic)
            let zr2 = ds_mul(zr, zr);       // zr²
            let zi2 = ds_mul(zi, zi);       // zi²
            let zri = ds_mul(zr, zi);       // zr * zi
            var two_zri = ds_add(zri, zri); // 2 * zr * zi

            // new_zr = zr² - zi² (before adding c)
            var diff = ds_add(zr2, vec2<f32>(-zi2.x, -zi2.y));

            // Tricorn (7): conjugate z before squaring → negate imaginary part
            if params.fractal_type == 7u {
                two_zri = vec2<f32>(-two_zri.x, -two_zri.y);
            }

            // Celtic (8): take abs of real part of z² (double-single abs: negate both hi and lo)
            if params.fractal_type == 8u {
                if diff.x < 0.0 { diff = vec2<f32>(-diff.x, -diff.y); }
            }

            // Perpendicular (9): new_zi = -2*|zr|*zi + ci
            // -2*|zr|*zi = -sign(zr) * 2*zr*zi = -sign(zr) * two_zri
            if params.fractal_type == 9u {
                if zr_sign >= 0.0 {
                    two_zri = vec2<f32>(-two_zri.x, -two_zri.y);
                }
                // else: zr was negative, |zr| = -zr, so -2*|zr|*zi = 2*zr*zi = two_zri (keep)
            }

            // Buffalo (10): real = abs(zr² - zi²), imag = -2*|zr|*|zi|
            if params.fractal_type == 10u {
                if diff.x < 0.0 { diff = vec2<f32>(-diff.x, -diff.y); }
                // Make two_zri negative: -|2*zr*zi| = -2*|zr|*|zi|
                if two_zri.x > 0.0 { two_zri = vec2<f32>(-two_zri.x, -two_zri.y); }
            }

            // new_zr = diff + cr, new_zi = two_zri + ci
            zr = ds_add(diff, cr);
            zi = ds_add(two_zri, ci);
        }

        // Escape test (f32 is sufficient for |z|² > 256)
        if zr.x * zr.x + zi.x * zi.x > escape_r2 {
            iter = i;
            break;
        }
    }

    // Smooth iteration count (f32 precision is fine here)
    // For power d: smooth = iter + 1 - log2(log|z|) / log2(d)
    var smooth_val: f32;
    if iter < max_i {
        let mag2 = zr.x * zr.x + zi.x * zi.x;
        let log_zn = log2(mag2) * 0.5;
        if params.fractal_type == 3u {
            smooth_val = f32(iter) + 1.0 - log2(log_zn) / log2(params.power);
        } else {
            smooth_val = f32(iter) + 1.0 - log2(log_zn);
        }
    } else {
        smooth_val = f32(max_i);
    }

    let iter_idx = params.sample_index * params.stride * params.resolution.y + idx;
    iterations[iter_idx] = smooth_val;
    final_z[idx] = vec2<f32>(zr.x, zi.x);
}
