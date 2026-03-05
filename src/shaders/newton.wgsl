// Newton/Nova fractal compute shader.
// Handles: Newton (4), Nova Julia (5), Nova Mandelbrot (6)
// Uses f(z) = z^n - 1 with configurable n.

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
    palette: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> iterations: array<f32>;
@group(0) @binding(2) var<storage, read_write> final_z: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> roots: array<vec2<f32>>;

// ── Double-single arithmetic (for coordinate mapping only) ──────────────────
// Each value is (hi, lo) where value = hi + lo (~48 bits of mantissa).

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return vec2<f32>(s, e);
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

fn ds_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    let e = a.y + b.y + s.y;
    return two_sum(s.x, e);
}

// ── Complex arithmetic ───────────────────────────────────────────────────────

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = b.x * b.x + b.y * b.y;
    if d < 1e-24 { return vec2<f32>(1e12, 0.0); }
    return vec2<f32>(
        (a.x * b.x + a.y * b.y) / d,
        (a.y * b.x - a.x * b.y) / d,
    );
}

fn cmag2(z: vec2<f32>) -> f32 {
    return z.x * z.x + z.y * z.y;
}

// z^n via repeated squaring (integer n).
fn cpow_int(z_in: vec2<f32>, n: u32) -> vec2<f32> {
    var result = vec2<f32>(1.0, 0.0);
    var base = z_in;
    var exp = n;
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

// Evaluate f(z) and f'(z) together, sharing the z^(n-1) computation.
// f(z) = z^n - 1, f'(z) = n * z^(n-1)
// Returns [f(z), f'(z)].
fn f_and_fp(z: vec2<f32>, n: u32) -> array<vec2<f32>, 2> {
    let zn1 = cpow_int(z, n - 1u);          // z^(n-1): one cpow_int call
    let zn = cmul(zn1, z);                   // z^n = z^(n-1) * z: one cmul
    let fz = zn - vec2<f32>(1.0, 0.0);
    let fpz = f32(n) * zn1;
    return array<vec2<f32>, 2>(fz, fpz);
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
    let n = params.num_roots; // polynomial degree
    let relax = params.relaxation;
    let max_i = params.max_iter;

    // Map pixel to complex plane using double-single precision for deep zoom
    let dx_pixels = f32(x) + params.sample_offset.x - f32(w - 1u) * 0.5;
    let dy_pixels = f32(y) + params.sample_offset.y - f32(h - 1u) * 0.5;
    let dx = two_prod(dx_pixels, params.pixel_step.x);
    let dy = two_prod(dy_pixels, params.pixel_step.y);
    let pixel_r = ds_add(vec2<f32>(params.center_hi.x, params.center_lo.x), dx);
    let pixel_i = ds_add(vec2<f32>(params.center_hi.y, params.center_lo.y), dy);
    let pixel = vec2<f32>(pixel_r.x, pixel_i.x);

    var z: vec2<f32>;
    var nova_c: vec2<f32>;

    // Set up based on fractal type
    switch params.fractal_type {
        case 5u: {
            // Nova Julia: z₀ = pixel, c = fixed constant
            z = pixel;
            nova_c = params.julia_c;
        }
        case 6u: {
            // Nova Mandelbrot: z₀ = critical point (1,0), c = pixel
            z = vec2<f32>(1.0, 0.0);
            nova_c = pixel;
        }
        default: {
            // Newton: z₀ = pixel, c = 0
            z = pixel;
            nova_c = vec2<f32>(0.0, 0.0);
        }
    }

    let tol: f32 = 1e-6;
    var iter: f32 = f32(max_i);

    // Pre-compute f(z) and f'(z) for initial z; cache across iterations.
    var ff = f_and_fp(z, n);
    var prev_step_mag2: f32 = 1e10;

    for (var i: u32 = 0u; i < max_i; i++) {
        let step_val = cdiv(ff[0], ff[1]);
        let z_new = z - relax * step_val + nova_c;

        // Convergence: |z_new - z|² < tol²
        // This works for BOTH Newton (c=0, converges to f(z)=0) and Nova
        // (c≠0, converges to a·f(z)/f'(z) = c, where f(z) ≠ 0).
        let delta = z_new - z;
        let step_mag2 = cmag2(delta);

        if step_mag2 < tol * tol {
            // Smooth iteration using consecutive step magnitudes
            if prev_step_mag2 > step_mag2 && prev_step_mag2 > 0.0 {
                iter = f32(i) + step_mag2 / prev_step_mag2;
            } else {
                iter = f32(i);
            }
            z = z_new;
            break;
        }

        prev_step_mag2 = step_mag2;
        z = z_new;
        // Cache f_and_fp for next iteration
        ff = f_and_fp(z_new, n);
    }

    iterations[idx] = iter;
    final_z[idx] = z;
}
