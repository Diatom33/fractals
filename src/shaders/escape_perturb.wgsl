// Perturbation-theory Mandelbrot compute shader.
// Uses a pre-computed reference orbit (arbitrary precision on CPU, stored as f32)
// and computes per-pixel perturbation deltas at f32 precision.
// Rebasing prevents glitches without multi-round rendering.

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

struct PerturbParams {
    ref_orbit_len: u32,     // number of valid entries in ref_orbit
    _pad: vec3<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> iterations: array<f32>;
@group(0) @binding(2) var<storage, read_write> final_z: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> ref_orbit: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> perturb: PerturbParams;

// Complex multiply
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.resolution.x;
    let h = params.resolution.y;
    if x >= w || y >= h { return; }

    let idx = y * params.stride + x;
    let max_i = params.max_iter;
    let ref_len = perturb.ref_orbit_len;

    // delta_c: pixel offset from reference point (center) in complex plane.
    // At deep zoom, pixel_step is tiny but that's fine — delta_c just needs
    // to represent the RELATIVE offset, which fits in f32.
    let dx_pixels = f32(x) + params.sample_offset.x - f32(w - 1u) * 0.5;
    let dy_pixels = f32(y) + params.sample_offset.y - f32(h - 1u) * 0.5;
    let dc = vec2<f32>(
        dx_pixels * params.pixel_step.x,
        dy_pixels * params.pixel_step.y,
    );

    // Perturbation iteration
    var dn = vec2<f32>(0.0, 0.0);  // delta_0 = 0 (both orbits start at z=0)
    var iter: u32 = max_i;
    var ref_i: u32 = 0u;           // current index into reference orbit

    for (var i: u32 = 0u; i < max_i; i++) {
        if ref_i >= ref_len { break; }

        let Zn = ref_orbit[ref_i];

        // delta_{n+1} = 2 * Z_n * delta_n + delta_n^2 + delta_c
        let two_Zn_dn = vec2<f32>(
            2.0 * (Zn.x * dn.x - Zn.y * dn.y),
            2.0 * (Zn.x * dn.y + Zn.y * dn.x),
        );
        let dn_sq = vec2<f32>(
            dn.x * dn.x - dn.y * dn.y,
            2.0 * dn.x * dn.y,
        );
        dn = two_Zn_dn + dn_sq + dc;
        ref_i += 1u;

        // Full orbit value: z_{n+1} = Z_{n+1} + delta_{n+1}
        // Use ref_orbit[ref_i] which is Z_{n+1} (we already incremented ref_i)
        var z_full: vec2<f32>;
        if ref_i < ref_len {
            z_full = ref_orbit[ref_i] + dn;
        } else {
            // Reference escaped; use last known Z + delta as approximation
            z_full = Zn + dn;
        }

        // Escape check
        let mag2 = dot(z_full, z_full);
        if mag2 > 256.0 {
            iter = i + 1u;
            break;
        }

        // Rebasing: if the full orbit passes near 0, the perturbation has
        // grown to nearly cancel the reference. Reset delta to the full value
        // and restart from the beginning of the reference orbit.
        // This prevents glitches without needing multiple reference orbits.
        let dn_mag2 = dot(dn, dn);
        if dot(z_full, z_full) < dn_mag2 {
            dn = z_full;
            ref_i = 0u;
        }
    }

    // Smooth iteration count
    var smooth_val: f32;
    if iter < max_i {
        var z_final: vec2<f32>;
        if ref_i < ref_len {
            z_final = ref_orbit[ref_i] + dn;
        } else {
            z_final = dn; // fallback
        }
        let mag2 = dot(z_final, z_final);
        let log_zn = log2(mag2) * 0.5;
        smooth_val = f32(iter) + 1.0 - log2(log_zn);
    } else {
        smooth_val = f32(max_i);
    }

    iterations[idx] = smooth_val;

    // Store final z for coloring
    if ref_i < ref_len {
        final_z[idx] = ref_orbit[ref_i] + dn;
    } else {
        final_z[idx] = dn;
    }
}
