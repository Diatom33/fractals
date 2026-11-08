// Median finalize shader.
// Reads all per-sample iteration values, finds the median of exterior samples,
// maps that single iteration count to a color, composites with interior coverage.
// Supports both escape-time and root-basin coloring (Newton/Nova).
//
// Palette functions live in palettes.wgsl (shared with colorize.wgsl),
// spliced in by src/shaders.rs at the marker below. Screen-space gradient
// palettes read sample slot 0's iteration landscape (sample_base = 0).

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
    sample_weight: f32,       // total_weight (set by host)
    stride: u32,
    palette: u32,
    sample_index: u32,
    num_samples: u32,
    coloring_param: f32,
    real_pixel_step: vec2<f32>,
    noise_seed: vec2<f32>,
    coloring_param_2: f32,
    pixel_step_log2: f32,     // log2 of the true pixel step (valid at any zoom depth)
    px_scale: f32,
    row_offset: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> iterations: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read> final_z: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> roots: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> orbit_traps: array<vec4<f32>>;

// Median of a pixel's exterior sub-sample iterations (max_iter if all interior).
// Palettes call this for neighbor reads (i = pixel index, sample_base is 0 here),
// so gradients and glow are computed on the same median field as the pixel itself.
fn iter_at(i: u32) -> f32 {
    let max_iter = f32(params.max_iter);
    let n = min(params.num_samples, 16u);
    let pps = params.stride * params.resolution.y;
    var v: array<f32, 16>;
    var c: u32 = 0u;
    for (var s: u32 = 0u; s < n; s++) {
        let it = iterations[s * pps + i];
        if it < max_iter {
            var j = c;
            loop {
                if j == 0u || v[j - 1u] <= it { break; }
                v[j] = v[j - 1u];
                j--;
            }
            v[j] = it;
            c++;
        }
    }
    if c == 0u { return max_iter; }
    if c % 2u == 1u { return v[c / 2u]; }
    return 0.5 * (v[c / 2u - 1u] + v[c / 2u]);
}

//INCLUDE:palettes

// Triangular-PDF dither of +-1 LSB before 8-bit quantization: dark smooth
// gradients otherwise show 1-level steps and streaks in print-size renders.
fn dither_hash(v: u32) -> f32 {
    var h = v * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return f32(h & 0xffffu) / 65535.0;
}

fn pack_rgba_at(rgb: vec3<f32>, x: u32, y: u32) -> u32 {
    let seed = x * 1973u + y * 9277u;
    let d = vec3<f32>(
        dither_hash(seed) + dither_hash(seed ^ 0x9e3779b9u) - 1.0,
        dither_hash(seed + 0x632be5abu) + dither_hash(seed ^ 0x85ebca6bu) - 1.0,
        dither_hash(seed + 0xc2b2ae35u) + dither_hash(seed ^ 0x27d4eb2fu) - 1.0,
    );
    let v = clamp(floor(rgb * 255.0 + 0.5 + d), vec3<f32>(0.0), vec3<f32>(255.0));
    return u32(v.x) | (u32(v.y) << 8u) | (u32(v.z) << 16u) | (255u << 24u);
}

// -- Main: median of exterior iterations, color, composite --------------------

// Max samples we support for median (limited by private array size in WGSL)
const MAX_SAMPLES: u32 = 64u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y + params.row_offset;
    let w = params.resolution.x;
    let h = params.resolution.y;
    if x >= w || y >= h { return; }

    let idx = y * params.stride + x;
    let max_iter = f32(params.max_iter);
    let n = min(params.num_samples, MAX_SAMPLES);
    let pixels_per_sample = params.stride * h;

    // Collect exterior iteration values
    var ext_iters: array<f32, 64>;
    var ext_count: u32 = 0u;

    for (var s: u32 = 0u; s < n; s++) {
        let iter_val = iterations[s * pixels_per_sample + idx];
        if iter_val < max_iter {
            ext_iters[ext_count] = iter_val;
            ext_count++;
        }
    }

    // Interior color: black for classic palettes, the palette's interior
    // material for palettes 11+.
    let fz_i = final_z[idx];
    var icolor = vec3<f32>(0.0);
    if params.color_mode == 0u && palette_has_interior(params.palette) {
        icolor = interior_color(fz_i.xy, idx);
    }

    // All interior
    if ext_count == 0u {
        output[idx] = pack_rgba_at(icolor, x, y);
        return;
    }

    // Insertion sort (fine for N ≤ 64)
    for (var i: u32 = 1u; i < ext_count; i++) {
        let key = ext_iters[i];
        var j: i32 = i32(i) - 1;
        loop {
            if j < 0 { break; }
            if ext_iters[u32(j)] <= key { break; }
            ext_iters[u32(j + 1)] = ext_iters[u32(j)];
            j--;
        }
        ext_iters[u32(j + 1)] = key;
    }

    // Median
    var median_iter: f32;
    if ext_count % 2u == 1u {
        median_iter = ext_iters[ext_count / 2u];
    } else {
        median_iter = (ext_iters[ext_count / 2u - 1u] + ext_iters[ext_count / 2u]) * 0.5;
    }

    // Color the median iteration value
    let z = fz_i.xy;
    let dz_log2 = fz_i.z;
    let dz_angle = fz_i.w;
    var color: vec3<f32>;
    if params.color_mode == 1u {
        color = basin_color(median_iter, z, params.num_roots);
    } else {
        color = escape_color(median_iter, z, dz_log2, dz_angle, x, y, 0u);
    }

    // Composite with interior coverage
    let ext_coverage = f32(ext_count) / f32(n);
    let final_color = mix(icolor, color, ext_coverage);

    output[idx] = pack_rgba_at(final_color, x, y);
}
