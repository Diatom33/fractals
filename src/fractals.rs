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
    pub fn to_gpu_params(&self, width: u32, height: u32) -> GpuParams {
        GpuParams {
            bounds: self.bounds,
            resolution: [width, height],
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
            _pad: [0; 2],
        }
    }
}

/// GPU-side uniform struct. Must match WGSL layout exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParams {
    pub bounds: [f32; 4],       // 16 bytes
    pub resolution: [u32; 2],   // 8 bytes
    pub max_iter: u32,          // 4 bytes
    pub fractal_type: u32,      // 4 bytes  (offset 24)
    pub julia_c: [f32; 2],      // 8 bytes  (offset 32)
    pub power: f32,             // 4 bytes  (offset 40)
    pub relaxation: f32,        // 4 bytes  (offset 44)
    pub color_mode: u32,        // 4 bytes  (offset 48)
    pub num_roots: u32,         // 4 bytes  (offset 52)
    pub _pad: [u32; 2],         // 8 bytes  (offset 56) → total 64 bytes
}
