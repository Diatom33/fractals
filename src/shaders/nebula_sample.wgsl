// Nebulabrot sampling compute shader.
// Each thread generates random c values, iterates z->z^2+c, and for escaping
// orbits, traces the trajectory and increments hit-count histograms.
// Three histograms with different iteration limits map to R, G, B channels.

struct NebulaParams {
    resolution: vec2<u32>,
    stride: u32,
    max_iter_r: u32,
    max_iter_g: u32,
    max_iter_b: u32,
    samples_per_thread: u32,
    dispatch_index: u32,
    sample_min: vec2<f32>,   // where to sample c from (full Mandelbrot region)
    sample_max: vec2<f32>,
    view_min: vec2<f32>,     // where to plot hits (export view bounds)
    view_max: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: NebulaParams;
@group(0) @binding(1) var<storage, read_write> histogram_r: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> histogram_g: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> histogram_b: array<atomic<u32>>;

fn pcg(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    return f32(pcg(state)) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var rng_state = gid.x * 1099087573u + params.dispatch_index * 2654435761u + 1u;

    let res_x = f32(params.resolution.x);
    let res_y = f32(params.resolution.y);
    let view_range = params.view_max - params.view_min;
    let sample_range = params.sample_max - params.sample_min;

    for (var s = 0u; s < params.samples_per_thread; s = s + 1u) {
        let cr = params.sample_min.x + rand_f32(&rng_state) * sample_range.x;
        let ci = params.sample_min.y + rand_f32(&rng_state) * sample_range.y;

        // First pass: iterate to check if c escapes, and find escape iteration
        var zr = 0.0f;
        var zi = 0.0f;
        var escaped = false;
        var escape_iter = params.max_iter_r;

        for (var i = 0u; i < params.max_iter_r; i = i + 1u) {
            let zr2 = zr * zr;
            let zi2 = zi * zi;
            if zr2 + zi2 > 4.0 {
                escape_iter = i;
                escaped = true;
                break;
            }
            let new_zr = zr2 - zi2 + cr;
            let new_zi = 2.0 * zr * zi + ci;
            zr = new_zr;
            zi = new_zi;
        }

        if !escaped {
            continue;
        }

        // Skip very short orbits (noise near the boundary)
        if escape_iter < 2u {
            continue;
        }

        // Second pass: re-iterate and plot each point into histograms
        zr = 0.0;
        zi = 0.0;
        for (var i = 0u; i < escape_iter; i = i + 1u) {
            let zr2 = zr * zr;
            let zi2 = zi * zi;
            let new_zr = zr2 - zi2 + cr;
            let new_zi = 2.0 * zr * zi + ci;
            zr = new_zr;
            zi = new_zi;

            // Map z to pixel coordinates in the view
            let tx = (zr - params.view_min.x) / view_range.x;
            let ty = (zi - params.view_min.y) / view_range.y;

            if tx >= 0.0 && tx < 1.0 && ty >= 0.0 && ty < 1.0 {
                let px = u32(tx * res_x);
                let py = u32(ty * res_y);
                let idx = py * params.stride + px;

                // Plot to B channel (lowest iter limit)
                if i < params.max_iter_b {
                    atomicAdd(&histogram_b[idx], 1u);
                }
                // Plot to G channel (medium iter limit)
                if i < params.max_iter_g {
                    atomicAdd(&histogram_g[idx], 1u);
                }
                // Plot to R channel (highest iter limit) — always true since i < escape_iter <= max_iter_r
                atomicAdd(&histogram_r[idx], 1u);
            }
        }
    }
}
