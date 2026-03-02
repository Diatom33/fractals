// Escape-time fractal compute shader.
// Handles: Mandelbrot (0), Julia (1), Burning Ship (2), Multibrot (3)

struct Params {
    bounds: vec4<f32>,       // x_min, x_max, y_min, y_max
    resolution: vec2<u32>,   // width, height
    max_iter: u32,
    fractal_type: u32,
    julia_c: vec2<f32>,
    power: f32,
    relaxation: f32,
    color_mode: u32,
    num_roots: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> iterations: array<f32>;
@group(0) @binding(2) var<storage, read_write> final_z: array<vec2<f32>>;

// ── Complex arithmetic ───────────────────────────────────────────────────────

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

    let idx = y * w + x;

    // Map pixel to complex plane
    let fx = f32(x) / f32(w - 1u);
    let fy = f32(y) / f32(h - 1u);
    let pixel = vec2<f32>(
        params.bounds.x + fx * (params.bounds.y - params.bounds.x),
        params.bounds.z + fy * (params.bounds.w - params.bounds.z),
    );

    var z: vec2<f32>;
    var c: vec2<f32>;

    // Set up initial conditions based on fractal type
    switch params.fractal_type {
        case 1u: {
            // Julia: z₀ = pixel, c = fixed
            z = pixel;
            c = params.julia_c;
        }
        default: {
            // Mandelbrot (0), Burning Ship (2), Multibrot (3): z₀ = 0, c = pixel
            z = vec2<f32>(0.0, 0.0);
            c = pixel;
        }
    }

    let escape_r2: f32 = 256.0; // escape radius squared (large for smooth_val coloring)
    let max_i = params.max_iter;
    var iter: u32 = max_i;

    for (var i: u32 = 0u; i < max_i; i++) {
        switch params.fractal_type {
            case 2u: {
                // Burning Ship: z = (|Re(z)| + i|Im(z)|)² + c
                let abs_z = vec2<f32>(abs(z.x), abs(z.y));
                z = csqr(abs_z) + c;
            }
            case 3u: {
                // Multibrot: z = z^d + c
                let d = u32(params.power);
                if d == 2u {
                    z = csqr(z) + c;
                } else {
                    z = cpow_int(z, d) + c;
                }
            }
            default: {
                // Mandelbrot / Julia: z = z² + c
                z = csqr(z) + c;
            }
        }

        if cmag2(z) > escape_r2 {
            iter = i;
            break;
        }
    }

    // Smooth iteration count for escaped points
    var smooth_val: f32;
    if iter < max_i {
        let log_zn = log2(cmag2(z)) * 0.5; // log2(|z|)
        let d = select(params.power, 2.0, params.fractal_type < 3u);
        smooth_val = f32(iter) + 1.0 - log2(log_zn) / log2(d);
    } else {
        smooth_val = f32(max_i);
    }

    iterations[idx] = smooth_val;
    final_z[idx] = z;
}
