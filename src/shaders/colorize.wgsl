// Colorization compute shader.
// Reads iteration data + final z, produces weighted color into accumulation buffer.
// Supports both escape-time and root-basin coloring with multiple palettes.
// Called once per sub-pixel sample; accumulates into a vec4<f32> buffer (rgb + weight).
//
// Two-population compositing: interior (max_iter) and exterior samples are tracked
// separately. accum.rgb holds weighted exterior color (in Oklab L,a,b), accum.w holds
// exterior weight only. Interior samples contribute nothing to accum — their coverage
// is inferred in finalize as (total_weight - exterior_weight) / total_weight.
//
// Palette functions live in palettes.wgsl (shared with median_finalize.wgsl),
// spliced in by src/shaders.rs at the marker below.

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
    palette: u32,
    sample_index: u32,
    num_samples: u32,
    coloring_param: f32,
    real_pixel_step: vec2<f32>,
    noise_seed: vec2<f32>,
    coloring_param_2: f32,
    _pad_128a: u32,
    _pad_128b: u32,
    _pad_128c: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> iterations: array<f32>;
@group(0) @binding(2) var<storage, read> final_z: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> roots: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> orbit_traps: array<vec4<f32>>;

//INCLUDE:palettes

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
    let sample_base = params.sample_index * params.stride * h;
    let smooth_iter = iterations[sample_base + idx];
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
        srgb_color = escape_color(smooth_iter, z, dz_mag, dz_angle, x, y, sample_base);
    } else {
        srgb_color = basin_color(smooth_iter, z, params.num_roots);
    }

    // Average in Oklab space for perceptual uniformity
    let oklab = srgb_to_oklab(srgb_color);

    // Accumulate: rgb = weighted Oklab (L,a,b), w = exterior weight only
    accum[idx] = prev + vec4<f32>(oklab * wt, wt);
}
