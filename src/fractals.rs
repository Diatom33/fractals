/// Fractal type definitions, parameters, and defaults.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractalType {
    Mandelbrot,
    Julia,
    BurningShip,
    Multibrot,
    Newton,
    NovaJulia,
    NovaMandelbrot,
}

impl FractalType {
    pub const ALL: &[FractalType] = &[
        FractalType::Mandelbrot,
        FractalType::Julia,
        FractalType::BurningShip,
        FractalType::Multibrot,
        FractalType::Newton,
        FractalType::NovaJulia,
        FractalType::NovaMandelbrot,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            FractalType::Mandelbrot => "Mandelbrot",
            FractalType::Julia => "Julia",
            FractalType::BurningShip => "Burning Ship",
            FractalType::Multibrot => "Multibrot",
            FractalType::Newton => "Newton",
            FractalType::NovaJulia => "Nova Julia",
            FractalType::NovaMandelbrot => "Nova Mandelbrot",
        }
    }

    /// GPU shader index matching WGSL fractal_type uniform.
    pub fn shader_index(&self) -> u32 {
        match self {
            FractalType::Mandelbrot => 0,
            FractalType::Julia => 1,
            FractalType::BurningShip => 2,
            FractalType::Multibrot => 3,
            FractalType::Newton => 4,
            FractalType::NovaJulia => 5,
            FractalType::NovaMandelbrot => 6,
        }
    }

    /// Whether this type uses escape-time (true) or root-basin (false) coloring.
    pub fn is_escape_time(&self) -> bool {
        matches!(
            self,
            FractalType::Mandelbrot
                | FractalType::Julia
                | FractalType::BurningShip
                | FractalType::Multibrot
        )
    }

    /// Whether this type needs the roots buffer for coloring.
    pub fn needs_roots(&self) -> bool {
        matches!(
            self,
            FractalType::Newton | FractalType::NovaJulia | FractalType::NovaMandelbrot
        )
    }

    pub fn default_bounds(&self) -> [f32; 4] {
        match self {
            FractalType::Mandelbrot => [-2.5, 1.0, -1.25, 1.25],
            FractalType::Julia => [-2.0, 2.0, -1.5, 1.5],
            FractalType::BurningShip => [-2.0, 1.5, -2.0, 0.5],
            FractalType::Multibrot => [-2.0, 2.0, -2.0, 2.0],
            FractalType::Newton => [-2.0, 2.0, -2.0, 2.0],
            FractalType::NovaJulia => [-2.0, 2.0, -2.0, 2.0],
            FractalType::NovaMandelbrot => [-2.0, 2.0, -2.0, 2.0],
        }
    }

    /// Which controls should be visible for this fractal type.
    pub fn visible_controls(&self) -> Controls {
        match self {
            FractalType::Mandelbrot | FractalType::BurningShip => Controls {
                power: false,
                julia_c: false,
                relaxation: false,
                poly_degree: false,
            },
            FractalType::Julia => Controls {
                power: false,
                julia_c: true,
                relaxation: false,
                poly_degree: false,
            },
            FractalType::Multibrot => Controls {
                power: true,
                julia_c: false,
                relaxation: false,
                poly_degree: false,
            },
            FractalType::Newton => Controls {
                power: false,
                julia_c: false,
                relaxation: true,
                poly_degree: true,
            },
            FractalType::NovaJulia => Controls {
                power: false,
                julia_c: true,
                relaxation: true,
                poly_degree: true,
            },
            FractalType::NovaMandelbrot => Controls {
                power: false,
                julia_c: false,
                relaxation: true,
                poly_degree: true,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Controls {
    pub power: bool,
    pub julia_c: bool,
    pub relaxation: bool,
    pub poly_degree: bool,
}

/// All parameters needed to render any fractal type.
#[derive(Debug, Clone)]
pub struct FractalParams {
    pub fractal_type: FractalType,
    pub bounds: [f32; 4], // x_min, x_max, y_min, y_max
    pub max_iter: u32,
    pub julia_c: [f32; 2],   // [re, im]
    pub power: f32,           // Multibrot exponent
    pub relaxation: f32,      // Nova relaxation parameter a
    pub poly_degree: u32,     // Newton/Nova polynomial degree n (for z^n - 1)
    pub supersampling: u32,   // 1 = off, 2 = 2x2, 3 = 3x3
}

impl Default for FractalParams {
    fn default() -> Self {
        Self {
            fractal_type: FractalType::Mandelbrot,
            bounds: FractalType::Mandelbrot.default_bounds(),
            max_iter: 100,
            julia_c: [-0.7, 0.27015],
            power: 2.0,
            relaxation: 1.0,
            poly_degree: 3,
            supersampling: 1,
        }
    }
}

impl FractalParams {
    /// Compute the nth roots of unity for z^n - 1 (Newton/Nova coloring).
    /// Returns Vec of [re, im] pairs.
    pub fn compute_roots(&self) -> Vec<[f32; 2]> {
        let n = self.poly_degree as usize;
        (0..n)
            .map(|k| {
                let angle = 2.0 * std::f32::consts::PI * (k as f32) / (n as f32);
                [angle.cos(), angle.sin()]
            })
            .collect()
    }

    /// Build the GPU uniform data (must match Params struct in WGSL).
    /// `display_w` and `display_h` are the visible display resolution.
    /// `stride` is the buffer row width in pixels (>= display_w, aligned for wgpu).
    /// Sub-pixel offsets for multi-pass supersampling are set separately.
    pub fn to_gpu_params(&self, display_w: u32, display_h: u32, stride: u32) -> GpuParams {
        GpuParams {
            bounds: self.bounds,
            resolution: [display_w, display_h],
            max_iter: self.max_iter,
            fractal_type: self.fractal_type.shader_index(),
            julia_c: self.julia_c,
            power: self.power,
            relaxation: self.relaxation,
            color_mode: if self.fractal_type.is_escape_time() { 0 } else { 1 },
            num_roots: if self.fractal_type.needs_roots() {
                self.poly_degree
            } else {
                0
            },
            sample_offset: [0.0, 0.0],
            sample_weight: 1.0,
            stride,
            _pad: [0; 2],
        }
    }
}

/// Mitchell-Netravali 1D filter (B=1/3, C=1/3).
/// Maps pixel distance to filter weight; supports negative lobes for sharpening.
pub fn mitchell_1d(x: f32) -> f32 {
    let b: f32 = 1.0 / 3.0;
    let c: f32 = 1.0 / 3.0;
    let ax = x.abs();
    if ax < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * ax * ax * ax
            + (-18.0 + 12.0 * b + 6.0 * c) * ax * ax
            + (6.0 - 2.0 * b))
            / 6.0
    } else if ax < 2.0 {
        ((-b - 6.0 * c) * ax * ax * ax
            + (6.0 * b + 30.0 * c) * ax * ax
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// Pre-compute sub-pixel sample positions and Mitchell-Netravali weights.
/// Returns (offset_x, offset_y, weight) tuples in pixel units.
/// Grid spans [-0.5, +0.5] pixels (endpoint-inclusive); filter radius = 0.75.
pub fn compute_samples(ss: u32) -> Vec<(f32, f32, f32)> {
    let radius: f32 = 0.75;
    let mut samples = Vec::with_capacity((ss * ss) as usize);
    for sy in 0..ss {
        for sx in 0..ss {
            // Endpoint-inclusive grid: ss=1 → 0.0, ss=2 → ±0.5, ss=3 → -0.5, 0.0, +0.5
            let offset_x = if ss == 1 {
                0.0
            } else {
                -0.5 + (sx as f32) / (ss as f32 - 1.0)
            };
            let offset_y = if ss == 1 {
                0.0
            } else {
                -0.5 + (sy as f32) / (ss as f32 - 1.0)
            };
            // Separable 2D filter: w(x,y) = w(x) * w(y), scaled to filter radius
            let wx = mitchell_1d(offset_x / radius * 2.0);
            let wy = mitchell_1d(offset_y / radius * 2.0);
            let w = wx * wy;
            samples.push((offset_x, offset_y, w));
        }
    }
    samples
}

/// GPU-side uniform struct. Must match WGSL layout exactly.
/// Total 80 bytes, aligned to 16 (vec4<f32> max alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParams {
    pub bounds: [f32; 4],           // 16 bytes (offset 0)
    pub resolution: [u32; 2],       // 8 bytes  (offset 16) — OUTPUT resolution
    pub max_iter: u32,              // 4 bytes  (offset 24)
    pub fractal_type: u32,          // 4 bytes  (offset 28)
    pub julia_c: [f32; 2],          // 8 bytes  (offset 32)
    pub power: f32,                 // 4 bytes  (offset 40)
    pub relaxation: f32,            // 4 bytes  (offset 44)
    pub color_mode: u32,            // 4 bytes  (offset 48)
    pub num_roots: u32,             // 4 bytes  (offset 52)
    pub sample_offset: [f32; 2],    // 8 bytes  (offset 56) — sub-pixel offset in pixel units
    pub sample_weight: f32,         // 4 bytes  (offset 64)
    pub stride: u32,                // 4 bytes  (offset 68) — buffer row stride in pixels (≥ resolution.x)
    pub _pad: [u32; 2],             // 8 bytes  (offset 72) → total 80 bytes
}
