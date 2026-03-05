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

    pub fn default_bounds(&self) -> [f64; 4] {
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
    pub bounds: [f64; 4], // x_min, x_max, y_min, y_max (f64 for zoom precision)
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
    /// Computes center (split into hi/lo f32 pair for double-single precision)
    /// and pixel_step from the f64 bounds.
    pub fn to_gpu_params(&self, display_w: u32, display_h: u32, stride: u32) -> GpuParams {
        let cx = (self.bounds[0] + self.bounds[1]) / 2.0;
        let cy = (self.bounds[2] + self.bounds[3]) / 2.0;
        let step_x = (self.bounds[1] - self.bounds[0]) / (display_w as f64 - 1.0).max(1.0);
        let step_y = (self.bounds[3] - self.bounds[2]) / (display_h as f64 - 1.0).max(1.0);

        // Split f64 center into f32 hi + f32 lo (Dekker splitting)
        let cx_hi = cx as f32;
        let cx_lo = (cx - cx_hi as f64) as f32;
        let cy_hi = cy as f32;
        let cy_lo = (cy - cy_hi as f64) as f32;

        GpuParams {
            center_hi: [cx_hi, cy_hi],
            center_lo: [cx_lo, cy_lo],
            pixel_step: [step_x as f32, step_y as f32],
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
        }
    }
}

/// Pre-compute sub-pixel sample positions and weights for anti-aliasing.
/// Returns (offset_x, offset_y, weight) tuples in pixel units.
///
/// Uses a uniform grid within the pixel footprint [-0.5, +0.5] with equal
/// weights (box filter SSAA). Fractals have infinite bandwidth so
/// reconstruction filters like Mitchell just add blur — simple averaging
/// of sub-pixel samples gives clean AA without softening.
///
/// Quality levels:
///   ss=1: Off (1 sample at center)
///   ss=2: 4x4 grid within pixel — 16 samples
///   ss=3: 8x8 grid within pixel — 64 samples
pub fn compute_samples(ss: u32) -> Vec<(f32, f32, f32)> {
    if ss <= 1 {
        return vec![(0.0, 0.0, 1.0)];
    }

    // Grid density within [-0.5, +0.5] pixel footprint
    let grid_n: u32 = match ss {
        2 => 4,  // 4x4 = 16 samples
        _ => 8,  // 8x8 = 64 samples
    };

    let mut samples = Vec::with_capacity((grid_n * grid_n) as usize);
    for sy in 0..grid_n {
        for sx in 0..grid_n {
            // Stratified grid: center of each sub-pixel cell
            let offset_x = -0.5 + (sx as f32 + 0.5) / grid_n as f32;
            let offset_y = -0.5 + (sy as f32 + 0.5) / grid_n as f32;
            samples.push((offset_x, offset_y, 1.0));
        }
    }
    samples
}

/// Perturbation-specific GPU uniform. Must match PerturbParams in escape_perturb.wgsl.
/// Kept separate from GpuParams to avoid changing all 4 shader files.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerturbGpuParams {
    pub ref_orbit_len: u32,
    pub pixel_step_exp: i32,
    pub _pad: [u32; 2],
}

/// Data for perturbation rendering: reference orbit + metadata.
pub struct PerturbData {
    /// Reference orbit Z_n values as f32 pairs, len = orbit_len.
    pub orbit: Vec<[f32; 2]>,
    /// How many iterations before reference escaped (or max_iter).
    pub orbit_len: u32,
}

/// Compute a reference orbit at arbitrary precision using rug (GMP/MPFR).
/// center_re, center_im are f64 view center; precision_bits scales with zoom depth.
pub fn compute_reference_orbit(
    center_re: f64,
    center_im: f64,
    max_iter: u32,
    pixel_step: f64,
) -> PerturbData {
    use rug::Float;

    // Precision: ~3.32 bits per decimal digit of zoom depth, plus safety margin
    let zoom_digits = if pixel_step > 0.0 {
        (-pixel_step.log10()).max(16.0) as u32
    } else {
        64
    };
    let precision = (zoom_digits * 4 + 64).max(128);

    let c_re = Float::with_val(precision, center_re);
    let c_im = Float::with_val(precision, center_im);
    let mut z_re = Float::with_val(precision, 0.0);
    let mut z_im = Float::with_val(precision, 0.0);

    let mut orbit = Vec::with_capacity(max_iter as usize + 1);
    let escape_r2 = 256.0_f64;

    for _ in 0..max_iter {
        // Store Z_n as f32 for GPU
        let zr_f32 = z_re.to_f32();
        let zi_f32 = z_im.to_f32();
        orbit.push([zr_f32, zi_f32]);

        // z = z^2 + c at arbitrary precision
        let zr2 = Float::with_val(precision, &z_re * &z_re);
        let zi2 = Float::with_val(precision, &z_im * &z_im);
        let zri = Float::with_val(precision, &z_re * &z_im);

        z_re = Float::with_val(precision, &zr2 - &zi2) + &c_re;
        z_im = Float::with_val(precision, &zri * 2u32) + &c_im;

        // Escape check
        let mag2: f64 = (Float::with_val(precision, &z_re * &z_re)
            + Float::with_val(precision, &z_im * &z_im))
        .to_f64();
        if mag2 > escape_r2 {
            // Store the escaping Z value too (needed for smooth coloring)
            orbit.push([z_re.to_f32(), z_im.to_f32()]);
            break;
        }
    }

    let orbit_len = orbit.len() as u32;
    PerturbData { orbit, orbit_len }
}

/// GPU-side uniform struct. Must match WGSL layout exactly.
/// Total 80 bytes. Uses center + pixel_step instead of raw bounds
/// to enable double-single (emulated f64) precision in shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParams {
    pub center_hi: [f32; 2],       // 8 bytes  (offset 0)  — view center (high part)
    pub center_lo: [f32; 2],       // 8 bytes  (offset 8)  — view center (low part, sub-ULP)
    pub pixel_step: [f32; 2],      // 8 bytes  (offset 16) — complex units per pixel
    pub resolution: [u32; 2],      // 8 bytes  (offset 24) — OUTPUT resolution
    pub max_iter: u32,             // 4 bytes  (offset 32)
    pub fractal_type: u32,         // 4 bytes  (offset 36)
    pub julia_c: [f32; 2],        // 8 bytes  (offset 40)
    pub power: f32,                // 4 bytes  (offset 48)
    pub relaxation: f32,           // 4 bytes  (offset 52)
    pub color_mode: u32,           // 4 bytes  (offset 56)
    pub num_roots: u32,            // 4 bytes  (offset 60)
    pub sample_offset: [f32; 2],   // 8 bytes  (offset 64) — sub-pixel offset in pixel units
    pub sample_weight: f32,        // 4 bytes  (offset 72)
    pub stride: u32,               // 4 bytes  (offset 76)
}
// Total: 80 bytes, no padding needed
