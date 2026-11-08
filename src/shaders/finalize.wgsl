// Finalize compute shader.
// Reads accumulated exterior color (Oklab L,a,b + exterior weight), computes
// interior coverage from total_weight, composites exterior color with black interior.

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
    sample_weight: f32,       // total_weight is passed here by the host
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
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn oklab_to_linear_srgb(L: f32, a_ok: f32, b_ok: f32) -> vec3<f32> {
    let l_ = L + 0.3963377774 * a_ok + 0.2158037573 * b_ok;
    let m_ = L - 0.1055613458 * a_ok - 0.0638541728 * b_ok;
    let s_ = L - 0.0894841775 * a_ok - 1.2914855480 * b_ok;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    let r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y + params.row_offset;
    let w = params.resolution.x;
    let h = params.resolution.y;
    if x >= w || y >= h { return; }

    let idx = y * params.stride + x;
    let a = accum[idx];
    let ext_weight = a.w;          // exterior samples' total weight
    let total_weight = params.sample_weight;  // total weight of all samples

    var color: vec3<f32>;
    if ext_weight > 0.0 {
        // Average exterior color in Oklab, convert to sRGB
        let avg_oklab = a.xyz / ext_weight;
        let linear = oklab_to_linear_srgb(avg_oklab.x, avg_oklab.y, avg_oklab.z);
        let ext_srgb = pow(linear, vec3<f32>(1.0 / 2.2));

        // Composite: exterior_color * exterior_coverage + black * interior_coverage
        let ext_coverage = ext_weight / total_weight;
        color = ext_srgb * ext_coverage;
    } else {
        // Pure interior pixel
        color = vec3<f32>(0.0);
    }

    output[idx] = pack_rgba_at(color, x, y);
}
