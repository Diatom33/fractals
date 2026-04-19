// Perturbation-theory compute shader for all z²+c escape-time fractals.
// Supports: Mandelbrot (0), Julia (1), Burning Ship (2), Tricorn (7),
//           Celtic (8), Perpendicular (9), Buffalo (10)
//
// Uses a pre-computed reference orbit (arbitrary precision on CPU, stored as
// double-single f32 pairs) and computes per-pixel perturbation deltas using
// DOUBLE-SINGLE mantissa * 2^exponent representation.
// Double-single δ doubles per-step precision from ~1e-7 (single f32) to ~1e-14
// (DS), pushing zoom depth from ~1e-40 to ~1e-80.
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
// A and B are DS-complex (re_hi, re_lo, im_hi, im_lo) so the BLA multiply
// preserves DS precision in δ. Without DS coefficients, multiplying DS δ by
// a single-f32 A throws away the lo bits of δ on every BLA step.
// Tree layout: bla[n * num_levels + level] covers iterations [n, n + 2^level).
struct BlaCoeff {
    a: vec4<f32>,        // 16 bytes: a_re_hi, a_re_lo, a_im_hi, a_im_lo
    b: vec4<f32>,        // 16 bytes: b_re_hi, b_re_lo, b_im_hi, b_im_lo
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

// ── Double-single (DS) primitive ops, using FMA to prevent compiler from
// ── splitting error-term arithmetic. Each "DS f32" value is (hi, lo) with
// ── |lo| ≤ ulp(hi)/2; together they give ~48 bits of mantissa precision.

fn fast_two_sum(a: f32, b: f32) -> vec2<f32> {
    // Requires |a| >= |b|. Cheaper than two_sum when invariant holds.
    let s = a + b;
    let err = b - (s - a);
    return vec2<f32>(s, err);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, err);
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
    // Exact a*b = p + err using FMA.
    let p = a * b;
    let err = fma(a, b, -p);
    return vec2<f32>(p, err);
}

// (a_hi + a_lo) + (b_hi + b_lo) → (s_hi, s_lo)
fn ds_add(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    let sh = two_sum(a_hi, b_hi);
    let sl = two_sum(a_lo, b_lo);
    let v = sh.y + sl.x;
    let h1 = fast_two_sum(sh.x, v);
    let v2 = h1.y + sl.y;
    return fast_two_sum(h1.x, v2);
}

// (a_hi + a_lo) * (b_hi + b_lo) → (p_hi, p_lo)
fn ds_mul(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    let p = two_prod(a_hi, b_hi);
    // Cross terms; a_lo*b_lo dropped (~ulp²·hi precision contribution).
    let lo = fma(a_hi, b_lo, fma(a_lo, b_hi, p.y));
    return fast_two_sum(p.x, lo);
}

// ── DS complex helpers. δ stored as (re_hi, re_lo, im_hi, im_lo).
// Single-precision results returned as vec4<f32> with same field order.

// (a_re + i a_im) * (b_re + i b_im)
fn dsc_mul(
    a_re_hi: f32, a_re_lo: f32, a_im_hi: f32, a_im_lo: f32,
    b_re_hi: f32, b_re_lo: f32, b_im_hi: f32, b_im_lo: f32,
) -> vec4<f32> {
    let rr = ds_mul(a_re_hi, a_re_lo, b_re_hi, b_re_lo);
    let ii = ds_mul(a_im_hi, a_im_lo, b_im_hi, b_im_lo);
    let re = ds_add(rr.x, rr.y, -ii.x, -ii.y);
    let ri = ds_mul(a_re_hi, a_re_lo, b_im_hi, b_im_lo);
    let ir = ds_mul(a_im_hi, a_im_lo, b_re_hi, b_re_lo);
    let im = ds_add(ri.x, ri.y, ir.x, ir.y);
    return vec4<f32>(re.x, re.y, im.x, im.y);
}

// (a_re + i a_im)²
fn dsc_sq(
    a_re_hi: f32, a_re_lo: f32, a_im_hi: f32, a_im_lo: f32,
) -> vec4<f32> {
    let rr = ds_mul(a_re_hi, a_re_lo, a_re_hi, a_re_lo);
    let ii = ds_mul(a_im_hi, a_im_lo, a_im_hi, a_im_lo);
    let re = ds_add(rr.x, rr.y, -ii.x, -ii.y);
    let ri = ds_mul(a_re_hi, a_re_lo, a_im_hi, a_im_lo);
    // im = 2 * ri. Renormalize after scaling to surface any carry into the lo.
    let im = fast_two_sum(2.0 * ri.x, 2.0 * ri.y);
    return vec4<f32>(re.x, re.y, im.x, im.y);
}

// Add two DS-complex values both at the same exponent.
fn dsc_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let r = ds_add(a.x, a.y, b.x, b.y);
    let i = ds_add(a.z, a.w, b.z, b.w);
    return vec4<f32>(r.x, r.y, i.x, i.y);
}

// Scale DS-complex by 2^e (exact when e small enough).
fn dsc_ldexp(a: vec4<f32>, e: i32) -> vec4<f32> {
    return vec4<f32>(ldexp(a.x, e), ldexp(a.y, e), ldexp(a.z, e), ldexp(a.w, e));
}

// Returns the shift that dsc_renorm would apply (so caller can update exp).
fn dsc_renorm_shift(a: vec4<f32>) -> i32 {
    let mag = max(abs(a.x), abs(a.z));
    if mag > 0.0 {
        return -i32(floor(log2(mag)));
    }
    return 0;
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

    // delta_c in extended-range DS: pixel_step is mantissa, pixel_step_exp is exponent.
    // Compute dc with DS precision: dx_pixels exact, dx*ps via two_prod.
    let dx_pixels = f32(x) + params.sample_offset.x - f32(w - 1u) * 0.5;
    let dy_pixels = f32(y) + params.sample_offset.y - f32(h - 1u) * 0.5;
    let dcx = two_prod(dx_pixels, params.pixel_step.x);
    let dcy = two_prod(dy_pixels, params.pixel_step.y);
    var dc = vec4<f32>(dcx.x, dcx.y, dcy.x, dcy.y);
    var dc_e = perturb.pixel_step_exp;
    // Normalize so max(|dc.re|, |dc.im|) is in [1, 2).
    let dc_shift = dsc_renorm_shift(dc);
    dc = dsc_ldexp(dc, dc_shift);
    dc_e = dc_e - dc_shift;

    let is_julia = ft == 1u;

    // Initial delta: Mandelbrot variants δ₀ = 0; Julia δ₀ = pixel offset.
    var dn: vec4<f32>;
    var dn_e: i32;
    if is_julia {
        dn = dc;
        dn_e = dc_e;
    } else {
        dn = vec4<f32>(0.0, 0.0, 0.0, 0.0);
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
            // log2(|δ|): mantissa hi is in [1, 2), so log2(|δ|) ≈ dn_e + log2(|dn.hi|).
            let dn_mag2 = dn.x * dn.x + dn.z * dn.z;
            if dn_mag2 > 0.0 {
                let delta_log2 = f32(dn_e) + 0.5 * log2(dn_mag2);
                var L = max_lvl;
                loop {
                    if L == 0u { break; }
                    L = L - 1u;
                    if L >= 31u { continue; }
                    let skip = 1u << L;
                    if (i + skip) > max_i { continue; }
                    if (ref_i + skip) > ref_len { continue; }
                    let node = bla_tree[ref_i * max_lvl + L];
                    if node.radius_log2 >= delta_log2 {
                        // Apply BLA: δ ← A·δ + B·δc with full DS×DS precision.
                        // node.a/b are DS-complex (re_hi, re_lo, im_hi, im_lo).
                        let a_dsc = dsc_mul(
                            node.a.x, node.a.y, node.a.z, node.a.w,
                            dn.x, dn.y, dn.z, dn.w,
                        );
                        let term_a_e = node.a_exp + dn_e;
                        let b_dsc = dsc_mul(
                            node.b.x, node.b.y, node.b.z, node.b.w,
                            dc.x, dc.y, dc.z, dc.w,
                        );
                        let term_b_e = node.b_exp + dc_e;
                        let e_max = max(term_a_e, term_b_e);
                        let a_aligned = dsc_ldexp(a_dsc, term_a_e - e_max);
                        let b_aligned = dsc_ldexp(b_dsc, term_b_e - e_max);
                        var sum = dsc_add(a_aligned, b_aligned);
                        let s = dsc_renorm_shift(sum);
                        sum = dsc_ldexp(sum, s);
                        dn = sum;
                        dn_e = e_max - s;
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

            // ── Pre-square modification on Z and δ ──────────────────
            // For Burning Ship and Tricorn, modify the operands by sign/conjugation.
            // δ pre-mods are applied to the mantissa (dn) directly.
            var Zeff_re_hi = Zn_re_hi;
            var Zeff_im_hi = Zn_im_hi;
            var Zeff_re_lo = Zn_re_lo;
            var Zeff_im_lo = Zn_im_lo;
            var deff = dn;

            if ft == 2u {
                // Burning Ship: sign of FULL z (not just reference Z) — needed near fold lines.
                let dn_re_real = ldexp(dn.x, dn_e);
                let dn_im_real = ldexp(dn.z, dn_e);
                let sr = signf(Zn_re_hi + dn_re_real);
                let si = signf(Zn_im_hi + dn_im_real);
                Zeff_re_hi = sr * Zn_re_hi;
                Zeff_im_hi = si * Zn_im_hi;
                Zeff_re_lo = sr * Zn_re_lo;
                Zeff_im_lo = si * Zn_im_lo;
                deff = vec4<f32>(sr * dn.x, sr * dn.y, si * dn.z, si * dn.w);
            } else if ft == 7u {
                // Tricorn: Z_eff = conj(Z), δ_eff = conj(δ) — negate imag parts.
                Zeff_im_hi = -Zn_im_hi;
                Zeff_im_lo = -Zn_im_lo;
                deff = vec4<f32>(dn.x, dn.y, -dn.z, -dn.w);
            }

            // Term 1: 2·Z_eff·δ_eff (DS×DS complex multiply).
            let t1_dsc_unscaled = dsc_mul(
                Zeff_re_hi, Zeff_re_lo, Zeff_im_hi, Zeff_im_lo,
                deff.x, deff.y, deff.z, deff.w,
            );
            // Multiply by 2 (exact)
            let t1 = vec4<f32>(2.0 * t1_dsc_unscaled.x, 2.0 * t1_dsc_unscaled.y,
                               2.0 * t1_dsc_unscaled.z, 2.0 * t1_dsc_unscaled.w);
            let t1_e = dn_e;

            // Term 2: δ_eff² (DS complex square).
            let t2 = dsc_sq(deff.x, deff.y, deff.z, deff.w);
            let t2_e = 2 * dn_e;

            // Sum terms (with or without δ_c).
            var e_new: i32;
            var sum: vec4<f32>;
            if is_julia {
                e_new = max(t1_e, t2_e);
                let s1 = dsc_ldexp(t1, t1_e - e_new);
                let s2 = dsc_ldexp(t2, t2_e - e_new);
                sum = dsc_add(s1, s2);
            } else {
                e_new = max(t1_e, max(t2_e, dc_e));
                let s1 = dsc_ldexp(t1, t1_e - e_new);
                let s2 = dsc_ldexp(t2, t2_e - e_new);
                let s3 = dsc_ldexp(dc, dc_e - e_new);
                sum = dsc_add(dsc_add(s1, s2), s3);
            }

            // Post-square modification (sign flips for Celtic/Perpendicular/Buffalo).
            if ft == 8u {
                let Zsq_re = Zn_re_hi * Zn_re_hi - Zn_im_hi * Zn_im_hi;
                let s = signf(Zsq_re);
                sum = vec4<f32>(s * sum.x, s * sum.y, sum.z, sum.w);
            } else if ft == 9u {
                let s = -signf(Zn_re_hi);
                sum = vec4<f32>(sum.x, sum.y, s * sum.z, s * sum.w);
            } else if ft == 10u {
                let Zsq_re = Zn_re_hi * Zn_re_hi - Zn_im_hi * Zn_im_hi;
                let Zsq_im = 2.0 * Zn_re_hi * Zn_im_hi;
                let sr = signf(Zsq_re);
                let si = -signf(Zsq_im);
                sum = vec4<f32>(sr * sum.x, sr * sum.y, si * sum.z, si * sum.w);
            }

            dn = sum;
            dn_e = e_new;
            // Renormalize mantissa.
            let s = dsc_renorm_shift(dn);
            dn = dsc_ldexp(dn, s);
            dn_e = dn_e - s;

            ref_i = ref_i + 1u;
            step = 1u;
        }

        // ── Common: z_full computation, traps, escape, rebase ─────────
        // Convert δ to single-precision real value for the escape/rebase tests.
        // (Z is bounded; once |z_full| > 16, we're escaping — full DS not needed here.)
        let dn_re_real = ldexp(dn.x, dn_e) + ldexp(dn.y, dn_e);
        let dn_im_real = ldexp(dn.z, dn_e) + ldexp(dn.w, dn_e);
        let dn_real = vec2<f32>(dn_re_real, dn_im_real);

        var z_full: vec2<f32>;
        if ref_i < ref_len {
            let Zn1 = ref_orbit[ref_i];
            z_full = vec2<f32>(Zn1.x, Zn1.y) + dn_real;
        } else {
            let Zn_last = ref_orbit[ref_len - 1u];
            z_full = vec2<f32>(Zn_last.x, Zn_last.y) + dn_real;
        }

        // Orbit trap distances (only on single-step iterations).
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

        // Rebase: when |z_full| < |delta|, glitch correction.
        let dn_mag2 = dot(dn_real, dn_real);
        if mag2 < dn_mag2 {
            dn = vec4<f32>(z_full.x, 0.0, z_full.y, 0.0);
            dn_e = 0;
            ref_i = 0u;
            let s = dsc_renorm_shift(dn);
            dn = dsc_ldexp(dn, s);
            dn_e = dn_e - s;
        }

        i = i + step;
    }

    // Smooth iteration count
    var smooth_val: f32;
    if iter < max_i {
        let dn_re_real = ldexp(dn.x, dn_e) + ldexp(dn.y, dn_e);
        let dn_im_real = ldexp(dn.z, dn_e) + ldexp(dn.w, dn_e);
        let dn_final = vec2<f32>(dn_re_real, dn_im_real);
        var z_final: vec2<f32>;
        if ref_i < ref_len {
            let Zn_f = ref_orbit[ref_i];
            z_final = vec2<f32>(Zn_f.x, Zn_f.y) + dn_final;
        } else {
            z_final = dn_final;
        }
        let m2 = dot(z_final, z_final);
        let log_zn = log2(m2) * 0.5;
        smooth_val = f32(iter) + 1.0 - log2(log_zn);
    } else {
        smooth_val = f32(max_i);
    }

    let iter_idx = params.sample_index * params.stride * params.resolution.y + idx;
    iterations[iter_idx] = smooth_val;

    let dn_out_re = ldexp(dn.x, dn_e) + ldexp(dn.y, dn_e);
    let dn_out_im = ldexp(dn.z, dn_e) + ldexp(dn.w, dn_e);
    let dn_out = vec2<f32>(dn_out_re, dn_out_im);
    if ref_i < ref_len {
        let Zn_o = ref_orbit[ref_i];
        let zf = vec2<f32>(Zn_o.x, Zn_o.y) + dn_out;
        final_z[idx] = vec4<f32>(zf.x, zf.y, 0.0, 0.0);
    } else {
        final_z[idx] = vec4<f32>(dn_out.x, dn_out.y, 0.0, 0.0);
    }
    orbit_traps[idx] = trap_min;
}
