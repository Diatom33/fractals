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
    coloring_param: f32,
    real_pixel_step: vec2<f32>,
    noise_seed: vec2<f32>,
}

struct PerturbParams {
    ref_orbit_len: u32,
    pixel_step_exp: i32,
    bla_num_levels: u32,    // 0 disables BLA (per-step perturbation only)
    _pad: u32,
}

// One BLA node: applies 2^level iterations as δ ← A·δ + B·δc.
// A and B mantissas are f32 with separate i32 exponents for extended range.
// Tree layout: bla[n * num_levels + level] covers iterations [n, n + 2^level).
struct BlaCoeff {
    a: vec2<f32>,
    b: vec2<f32>,
    a_exp: i32,
    b_exp: i32,
    radius_log2: f32,    // node valid when log2(|δ|) ≤ radius_log2
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> iterations: array<f32>;
@group(0) @binding(2) var<storage, read_write> final_z: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ref_orbit: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> perturb: PerturbParams;
@group(0) @binding(5) var<storage, read_write> orbit_traps: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> bla_tree: array<BlaCoeff>;

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
    // Orbit trap points for Canopy palette
    let trap0 = vec2<f32>(0.0, 0.0);
    let trap1 = vec2<f32>(1.0, 0.0);
    let trap2 = vec2<f32>(-0.5, 0.866);
    let trap3 = vec2<f32>(-0.5, -0.866);
    var trap_min = vec4<f32>(1e20, 1e20, 1e20, 1e20);

    var iter: u32 = max_i;
    var ref_i: u32 = 0u;
    var i: u32 = 0u;

    let bla_enabled = perturb.bla_num_levels > 0u && ft == 0u;
    let max_lvl = perturb.bla_num_levels;

    loop {
        if i >= max_i { break; }
        if ref_i >= ref_len { break; }

        // ── Try BLA stepping (Mandelbrot only) ────────────────────────
        var step: u32 = 0u;
        if bla_enabled && ref_i > 0u {
            // log2(|δ|): dn_m is normalized so |dn_m| ∈ [1, 2). Use exact.
            let dn_mag2 = dn_m.x * dn_m.x + dn_m.y * dn_m.y;
            if dn_mag2 > 0.0 {
                let delta_log2 = f32(dn_e) + 0.5 * log2(dn_mag2);
                // Walk levels from highest down; first valid wins.
                var L = max_lvl;
                loop {
                    if L == 0u { break; }
                    L = L - 1u;
                    if L >= 31u { continue; }     // shift safety
                    let skip = 1u << L;
                    if (i + skip) > max_i { continue; }
                    if (ref_i + skip) > ref_len { continue; }
                    let node = bla_tree[ref_i * max_lvl + L];
                    if node.radius_log2 >= delta_log2 {
                        // Apply BLA: δ ← A·δ + B·δc (extended-range complex math).
                        let term_a_m = vec2<f32>(
                            node.a.x * dn_m.x - node.a.y * dn_m.y,
                            node.a.x * dn_m.y + node.a.y * dn_m.x,
                        );
                        let term_a_e = node.a_exp + dn_e;
                        let term_b_m = vec2<f32>(
                            node.b.x * dc_m.x - node.b.y * dc_m.y,
                            node.b.x * dc_m.y + node.b.y * dc_m.x,
                        );
                        let term_b_e = node.b_exp + dc_e;
                        let e_max = max(term_a_e, term_b_e);
                        dn_m = ldexp_v2(term_a_m, term_a_e - e_max)
                             + ldexp_v2(term_b_m, term_b_e - e_max);
                        dn_e = e_max;
                        // Renormalize
                        let mag = max(abs(dn_m.x), abs(dn_m.y));
                        if mag > 0.0 {
                            let shift_val = i32(floor(log2(mag)));
                            dn_m = ldexp_v2(dn_m, -shift_val);
                            dn_e = dn_e + shift_val;
                        }
                        ref_i = ref_i + skip;
                        step = skip;
                        break;
                    }
                }
            }
        }

        // ── Single-step perturbation ──────────────────────────────────
        var Zn_re_hi: f32 = 0.0;
        var Zn_im_hi: f32 = 0.0;
        if step == 0u {
            let Zn = ref_orbit[ref_i];
            Zn_re_hi = Zn.x;
            Zn_im_hi = Zn.y;
            let Zn_re_lo = Zn.z;
            let Zn_im_lo = Zn.w;

            // ── Pre-square modification ─────────────────────────────
            var Zeff_re_hi = Zn_re_hi;
            var Zeff_im_hi = Zn_im_hi;
            var Zeff_re_lo = Zn_re_lo;
            var Zeff_im_lo = Zn_im_lo;
            var deff_m = dn_m;

            if ft == 2u {
                let dn_real_pre = ldexp_v2(dn_m, dn_e);
                let sr = signf(Zn_re_hi + dn_real_pre.x);
                let si = signf(Zn_im_hi + dn_real_pre.y);
                Zeff_re_hi = sr * Zn_re_hi;
                Zeff_im_hi = si * Zn_im_hi;
                Zeff_re_lo = sr * Zn_re_lo;
                Zeff_im_lo = si * Zn_im_lo;
                deff_m = vec2<f32>(sr * dn_m.x, si * dn_m.y);
            } else if ft == 7u {
                Zeff_im_hi = -Zn_im_hi;
                Zeff_im_lo = -Zn_im_lo;
                deff_m = vec2<f32>(dn_m.x, -dn_m.y);
            }

            // 2·Z_eff·δ_eff
            let a = fma(Zeff_re_hi, deff_m.x, Zeff_re_lo * deff_m.x);
            let b = fma(Zeff_im_hi, deff_m.y, Zeff_im_lo * deff_m.y);
            let c = fma(Zeff_re_hi, deff_m.y, Zeff_re_lo * deff_m.y);
            let d = fma(Zeff_im_hi, deff_m.x, Zeff_im_lo * deff_m.x);
            let t1 = vec2<f32>(2.0 * (a - b), 2.0 * (c + d));
            let t1_e = dn_e;

            let t2 = vec2<f32>(
                deff_m.x * deff_m.x - deff_m.y * deff_m.y,
                2.0 * deff_m.x * deff_m.y,
            );
            let t2_e = 2 * dn_e;

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

            // Post-square modification
            if ft == 8u {
                let Zsq_re = Zn_re_hi * Zn_re_hi - Zn_im_hi * Zn_im_hi;
                sum.x = sum.x * signf(Zsq_re);
            } else if ft == 9u {
                sum.y = sum.y * (-signf(Zn_re_hi));
            } else if ft == 10u {
                let Zsq_re = Zn_re_hi * Zn_re_hi - Zn_im_hi * Zn_im_hi;
                let Zsq_im = 2.0 * Zn_re_hi * Zn_im_hi;
                sum.x = sum.x * signf(Zsq_re);
                sum.y = sum.y * (-signf(Zsq_im));
            }

            dn_m = sum;
            dn_e = e_new;

            // Renormalize
            let mag = max(abs(dn_m.x), abs(dn_m.y));
            if mag > 0.0 {
                let shift_val = i32(floor(log2(mag)));
                dn_m = ldexp_v2(dn_m, -shift_val);
                dn_e = dn_e + shift_val;
            }

            ref_i = ref_i + 1u;
            step = 1u;
        }

        // ── Common: z_full computation, traps, escape, rebase ─────────
        let dn_real = ldexp_v2(dn_m, dn_e);
        var z_full: vec2<f32>;
        if ref_i < ref_len {
            let Zn1 = ref_orbit[ref_i];
            z_full = vec2<f32>(Zn1.x, Zn1.y) + dn_real;
        } else {
            // BLA stepped exactly to end; use last orbit point
            let Zn_last = ref_orbit[ref_len - 1u];
            z_full = vec2<f32>(Zn_last.x, Zn_last.y) + dn_real;
        }

        // Orbit trap distances (only sampled on single-step iterations to avoid
        // skipping minima inside a BLA chunk; BLA is for non-Canopy palettes anyway)
        if step == 1u {
            trap_min.x = min(trap_min.x, length(z_full - trap0));
            trap_min.y = min(trap_min.y, length(z_full - trap1));
            trap_min.z = min(trap_min.z, length(z_full - trap2));
            trap_min.w = min(trap_min.w, length(z_full - trap3));
        }

        // Escape check
        let mag2 = dot(z_full, z_full);
        if mag2 > 256.0 {
            iter = i + step;
            break;
        }

        // Rebase: when |z_full| < |delta|, glitch correction kicks in.
        let dn_mag2 = dot(dn_real, dn_real);
        if mag2 < dn_mag2 {
            dn_m = z_full;
            dn_e = 0;
            ref_i = 0u;
            let mag = max(abs(dn_m.x), abs(dn_m.y));
            if mag > 0.0 {
                let shift_val = i32(floor(log2(mag)));
                dn_m = ldexp_v2(dn_m, -shift_val);
                dn_e = dn_e + shift_val;
            }
        }

        i = i + step;
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

    // Store final z for coloring (dz_mag=0 — derivative tracking not available in perturbation mode)
    let dn_out = ldexp_v2(dn_m, dn_e);
    if ref_i < ref_len {
        let Zn_o = ref_orbit[ref_i];
        let zf = vec2<f32>(Zn_o.x, Zn_o.y) + dn_out;
        final_z[idx] = vec4<f32>(zf.x, zf.y, 0.0, 0.0);
    } else {
        final_z[idx] = vec4<f32>(dn_out.x, dn_out.y, 0.0, 0.0);
    }
    orbit_traps[idx] = trap_min;
}
