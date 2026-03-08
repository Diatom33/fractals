// Nebulabrot finalize compute shader.
// Reads three histogram buffers (R, G, B), applies cbrt normalization
// with percentile-based exposure, and packs to RGBA output buffer.

struct NebFinParams {
    resolution: vec2<u32>,
    stride: u32,
    max_r: u32,
    max_g: u32,
    max_b: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: NebFinParams;
@group(0) @binding(1) var<storage, read> histogram_r: array<u32>;
@group(0) @binding(2) var<storage, read> histogram_g: array<u32>;
@group(0) @binding(3) var<storage, read> histogram_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;

fn pack_rgba(rgb: vec3<f32>) -> u32 {
    let r = u32(clamp(rgb.x * 255.0, 0.0, 255.0));
    let g = u32(clamp(rgb.y * 255.0, 0.0, 255.0));
    let b = u32(clamp(rgb.z * 255.0, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.resolution.x || y >= params.resolution.y { return; }

    let idx = y * params.stride + x;

    let count_r = f32(histogram_r[idx]);
    let count_g = f32(histogram_g[idx]);
    let count_b = f32(histogram_b[idx]);

    var r = 0.0f;
    var g = 0.0f;
    var b = 0.0f;

    // cbrt normalization: pow(count/max, 1/3), clamped to [0,1]
    // max values are precomputed percentile-based exposure on CPU
    if params.max_r > 0u {
        r = clamp(pow(count_r / f32(params.max_r), 1.0 / 3.0), 0.0, 1.0);
    }
    if params.max_g > 0u {
        g = clamp(pow(count_g / f32(params.max_g), 1.0 / 3.0), 0.0, 1.0);
    }
    if params.max_b > 0u {
        b = clamp(pow(count_b / f32(params.max_b), 1.0 / 3.0), 0.0, 1.0);
    }

    output[idx] = pack_rgba(vec3<f32>(r, g, b));
}
