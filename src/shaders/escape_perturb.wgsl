// Perturbation-theory compute shader for all z²+c escape-time fractals.
// Supports: Mandelbrot (0), Julia (1), Burning Ship (2), Tricorn (7),
//           Celtic (8), Perpendicular (9), Buffalo (10)
//
// Uses a pre-computed reference orbit (arbitrary precision on CPU, stored as
// double-single f32 pairs) and computes per-pixel perturbation deltas using
// mantissa * 2^exponent representation.
// This allows zoom depths far beyond f32's ~1e-38 limit (to 1e-300+).
// Rebasing prevents glitches without multi-round rendering.
//
// All variants decompose into:
//   1. Pre-square modification (abs for Burning Ship, conj for Tricorn)
//   2. Standard perturbation: 2·Z_eff·δ_eff + δ_eff²
//   3. Post-square modification (abs(Re) for Celtic, sign flips for others)
//   4. Add δ_c
//
// Mandelbrot: z₀=0, c=pixel → δ_c=pixel_offset, δ_z₀=0
// Julia:      z₀=pixel, c=fixed → δ_c=0, δ_z₀=pixel_offset

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
    sample_index: u32,
    num_samples: u32,
    _pad: u32,
}

struct PerturbParams {
    ref_orbit_len: u32,
    pixel_step_exp: i32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> iterations: array<f32>;
@group(0) @binding(2) var<storage, read_write> final_z: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> ref_orbit: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> perturb: PerturbParams;

// Complex multiply
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ldexp for vec2
fn ldexp_v2(v: vec2<f32>, e: i32) -> vec2<f32> {
    return vec2<f32>(ldexp(v.x, e), ldexp(v.y, e));
}

// Sign function: returns 1.0 for x >= 0, -1.0 for x < 0
fn signf(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
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
    let ft = params.fractal_type;

    // delta_c in extended-range: pixel_step is mantissa, pixel_step_exp is exponent
    let dx_pixels = f32(x) + params.sample_offset.x - f32(w - 1u) * 0.5;
    let dy_pixels = f32(y) + params.sample_offset.y - f32(h - 1u) * 0.5;
    let dc_raw = vec2<f32>(
        dx_pixels * params.pixel_step.x,
        dy_pixels * params.pixel_step.y,
    );
    // Normalize dc mantissa
    var dc_m = dc_raw;
    var dc_e = perturb.pixel_step_exp;
    let dc_mag = max(abs(dc_m.x), abs(dc_m.y));
    if dc_mag > 0.0 {
        let dc_shift = i32(floor(log2(dc_mag)));
        dc_m = ldexp_v2(dc_m, -dc_shift);
        dc_e = dc_e + dc_shift;
    }

    let is_julia = ft == 1u;

    // Initial delta: Mandelbrot variants: δ_z₀=0, Julia: δ_z₀=pixel_offset
    var dn_m: vec2<f32>;
    var dn_e: i32;
    if is_julia {
        dn_m = dc_m;
        dn_e = dc_e;
    } else {
        dn_m = vec2<f32>(0.0, 0.0);
        dn_e = 0;
    }
    var iter: u32 = max_i;
    var ref_i: u32 = 0u;

    for (var i: u32 = 0u; i < max_i; i++) {
        if ref_i >= ref_len { break; }

        // Read Z_n as double-single: (re_hi, im_hi, re_lo, im_lo)
        let Zn = ref_orbit[ref_i];
        let Zn_re_hi = Zn.x;
        let Zn_im_hi = Zn.y;
        let Zn_re_lo = Zn.z;
        let Zn_im_lo = Zn.w;

        // ── Pre-square modification ─────────────────────────────────
        // Compute Z_eff and δ_eff based on fractal type.
        // The perturbation formula is: δ' = 2·Z_eff·δ_eff + δ_eff² + δ_c
        var Zeff_re_hi = Zn_re_hi;
        var Zeff_im_hi = Zn_im_hi;
        var Zeff_re_lo = Zn_re_lo;
        var Zeff_im_lo = Zn_im_lo;
        var deff_m = dn_m;

        if ft == 2u {
            // Burning Ship: Z_eff = (|Re(Z)|, |Im(Z)|), δ_eff sign-adjusted
            let sr = signf(Zn_re_hi);
            let si = signf(Zn_im_hi);
            Zeff_re_hi = abs(Zn_re_hi);
            Zeff_im_hi = abs(Zn_im_hi);
            Zeff_re_lo = sr * Zn_re_lo;
            Zeff_im_lo = si * Zn_im_lo;
            deff_m = vec2<f32>(sr * dn_m.x, si * dn_m.y);
        } else if ft == 7u {
            // Tricorn: Z_eff = conj(Z), δ_eff = conj(δ)
            Zeff_im_hi = -Zn_im_hi;
            Zeff_im_lo = -Zn_im_lo;
            deff_m = vec2<f32>(dn_m.x, -dn_m.y);
        }

        // ── Standard perturbation: 2·Z_eff·δ_eff + δ_eff² ─────────
        // Term 1: 2·Z_eff·δ_eff (complex multiply with double-single Z_eff)
        let a = fma(Zeff_re_hi, deff_m.x, Zeff_re_lo * deff_m.x);
        let b = fma(Zeff_im_hi, deff_m.y, Zeff_im_lo * deff_m.y);
        let c = fma(Zeff_re_hi, deff_m.y, Zeff_re_lo * deff_m.y);
        let d = fma(Zeff_im_hi, deff_m.x, Zeff_im_lo * deff_m.x);
        let t1 = vec2<f32>(2.0 * (a - b), 2.0 * (c + d));
        let t1_e = dn_e;

        // Term 2: δ_eff² (complex square of delta mantissa)
        let t2 = vec2<f32>(
            deff_m.x * deff_m.x - deff_m.y * deff_m.y,
            2.0 * deff_m.x * deff_m.y,
        );
        let t2_e = 2 * dn_e;

        // Sum terms (with or without δ_c)
        var e_new: i32;
        var sum: vec2<f32>;
        if is_julia {
            e_new = max(t1_e, t2_e);
            let s1 = ldexp_v2(t1, t1_e - e_new);
            let s2 = ldexp_v2(t2, t2_e - e_new);
            sum = s1 + s2;
        } else {
            let t3 = dc_m;
            let t3_e = dc_e;
            e_new = max(t1_e, max(t2_e, t3_e));
            let s1 = ldexp_v2(t1, t1_e - e_new);
            let s2 = ldexp_v2(t2, t2_e - e_new);
            let s3 = ldexp_v2(t3, t3_e - e_new);
            sum = s1 + s2 + s3;
        }

        // ── Post-square modification ────────────────────────────────
        // Adjust the result based on fractal type.
        // These variants modify z² (not z) before adding c.
        if ft == 8u {
            // Celtic: |Re(z²)| → multiply Re by sign(Re(Z²))
            let Zsq_re = Zn_re_hi * Zn_re_hi - Zn_im_hi * Zn_im_hi;
            sum.x = sum.x * signf(Zsq_re);
        } else if ft == 9u {
            // Perpendicular: Im = -2|Re(z)|·Im(z) → multiply Im by -sign(Re(Z))
            sum.y = sum.y * (-signf(Zn_re_hi));
        } else if ft == 10u {
            // Buffalo: |Re(z²)|, -|Im(z²)| → adjust both
            let Zsq_re = Zn_re_hi * Zn_re_hi - Zn_im_hi * Zn_im_hi;
            let Zsq_im = 2.0 * Zn_re_hi * Zn_im_hi;
            sum.x = sum.x * signf(Zsq_re);
            sum.y = sum.y * (-signf(Zsq_im));
        }

        dn_m = sum;
        dn_e = e_new;

        // Renormalize mantissa
        let mag = max(abs(dn_m.x), abs(dn_m.y));
        if mag > 0.0 {
            let shift_val = i32(floor(log2(mag)));
            dn_m = ldexp_v2(dn_m, -shift_val);
            dn_e = dn_e + shift_val;
        }

        ref_i += 1u;

        // Full orbit value: z_{n+1} = Z_{n+1} + delta_{n+1}
        let dn_real = ldexp_v2(dn_m, dn_e);
        var z_full: vec2<f32>;
        if ref_i < ref_len {
            let Zn1 = ref_orbit[ref_i];
            z_full = vec2<f32>(Zn1.x, Zn1.y) + dn_real;
        } else {
            z_full = vec2<f32>(Zn_re_hi, Zn_im_hi) + dn_real;
        }

        // Escape check
        let mag2 = dot(z_full, z_full);
        if mag2 > 256.0 {
            iter = i + 1u;
            break;
        }

        // Rebasing: if |z_full| < |delta|, reset delta to z_full
        let dn_mag2 = dot(dn_real, dn_real);
        if dot(z_full, z_full) < dn_mag2 {
            dn_m = z_full;
            dn_e = 0;
            ref_i = 0u;
        }
    }

    // Smooth iteration count
    var smooth_val: f32;
    if iter < max_i {
        let dn_final = ldexp_v2(dn_m, dn_e);
        var z_final: vec2<f32>;
        if ref_i < ref_len {
            let Zn_f = ref_orbit[ref_i];
            z_final = vec2<f32>(Zn_f.x, Zn_f.y) + dn_final;
        } else {
            z_final = dn_final;
        }
        let mag2 = dot(z_final, z_final);
        let log_zn = log2(mag2) * 0.5;
        smooth_val = f32(iter) + 1.0 - log2(log_zn);
    } else {
        smooth_val = f32(max_i);
    }

    let iter_idx = params.sample_index * params.stride * params.resolution.y + idx;
    iterations[iter_idx] = smooth_val;

    // Store final z for coloring
    let dn_out = ldexp_v2(dn_m, dn_e);
    if ref_i < ref_len {
        let Zn_o = ref_orbit[ref_i];
        final_z[idx] = vec2<f32>(Zn_o.x, Zn_o.y) + dn_out;
    } else {
        final_z[idx] = dn_out;
    }
}
