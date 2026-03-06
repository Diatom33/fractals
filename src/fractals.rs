/// Fractal type definitions, parameters, and defaults.

use rug::{Assign, Float};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractalType {
    Mandelbrot,
    Julia,
    BurningShip,
    Multibrot,
    Newton,
    NovaJulia,
    NovaMandelbrot,
    Tricorn,
    Celtic,
    Perpendicular,
    Buffalo,
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
        FractalType::Tricorn,
        FractalType::Celtic,
        FractalType::Perpendicular,
        FractalType::Buffalo,
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
            FractalType::Tricorn => "Tricorn",
            FractalType::Celtic => "Celtic",
            FractalType::Perpendicular => "Perpendicular",
            FractalType::Buffalo => "Buffalo",
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
            FractalType::Tricorn => 7,
            FractalType::Celtic => 8,
            FractalType::Perpendicular => 9,
            FractalType::Buffalo => 10,
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
                | FractalType::Tricorn
                | FractalType::Celtic
                | FractalType::Perpendicular
                | FractalType::Buffalo
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
            FractalType::Tricorn => [-2.5, 1.5, -2.0, 2.0],
            FractalType::Celtic => [-2.5, 1.5, -2.0, 2.0],
            FractalType::Perpendicular => [-2.5, 1.5, -2.0, 2.0],
            FractalType::Buffalo => [-2.5, 1.5, -2.0, 2.0],
        }
    }

    /// Which controls should be visible for this fractal type.
    pub fn visible_controls(&self) -> Controls {
        match self {
            FractalType::Mandelbrot | FractalType::BurningShip | FractalType::Tricorn | FractalType::Celtic | FractalType::Perpendicular | FractalType::Buffalo => Controls {
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
/// View is represented as center (arbitrary-precision via rug) + half-range (f64).
/// This allows deep zoom to 1e-300+ since the center maintains full precision
/// and half_range only needs relative accuracy.
#[derive(Debug, Clone)]
pub struct FractalParams {
    pub fractal_type: FractalType,
    pub center_re: Float,     // arbitrary precision center (real)
    pub center_im: Float,     // arbitrary precision center (imaginary)
    pub half_range_x: f64,    // half the x extent in complex units
    pub half_range_y: f64,    // half the y extent in complex units
    pub max_iter: u32,
    pub julia_c: [f32; 2],   // [re, im]
    pub power: f32,           // Multibrot exponent
    pub relaxation: f32,      // Nova relaxation parameter a
    pub poly_degree: u32,     // Newton/Nova polynomial degree n (for z^n - 1)
    pub supersampling: u32,   // 1 = off, 2 = 2x2, 3 = 3x3
    pub palette: ColorPalette,
    pub use_median: bool,     // true = median iteration SS, false = Oklab accumulation SS
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPalette {
    Classic,     // 0: HSV oscillating hue (original)
    Oklab,       // 1: Perceptually uniform lightness, varying hue
    Smooth,      // 2: iq-style cosine gradient
    Monochrome,  // 3: Single hue (blue), varying lightness
}

impl ColorPalette {
    pub const ALL: &[ColorPalette] = &[
        ColorPalette::Classic,
        ColorPalette::Oklab,
        ColorPalette::Smooth,
        ColorPalette::Monochrome,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ColorPalette::Classic => "Classic HSV",
            ColorPalette::Oklab => "Oklab Uniform",
            ColorPalette::Smooth => "Smooth Gradient",
            ColorPalette::Monochrome => "Monochrome",
        }
    }

    pub fn shader_index(&self) -> u32 {
        match self {
            ColorPalette::Classic => 0,
            ColorPalette::Oklab => 1,
            ColorPalette::Smooth => 2,
            ColorPalette::Monochrome => 3,
        }
    }
}

impl Default for FractalParams {
    fn default() -> Self {
        let b = FractalType::Mandelbrot.default_bounds();
        Self {
            fractal_type: FractalType::Mandelbrot,
            center_re: Float::with_val(128, (b[0] + b[1]) / 2.0),
            center_im: Float::with_val(128, (b[2] + b[3]) / 2.0),
            half_range_x: (b[1] - b[0]) / 2.0,
            half_range_y: (b[3] - b[2]) / 2.0,
            max_iter: 100,
            julia_c: [-0.7, 0.27015],
            power: 2.0,
            relaxation: 1.0,
            poly_degree: 3,
            supersampling: 1,
            palette: ColorPalette::Oklab,
            use_median: true,
        }
    }
}

impl FractalParams {
    /// Derive f64 bounds for display and legacy compatibility.
    pub fn bounds_f64(&self) -> [f64; 4] {
        let cx = self.center_re.to_f64();
        let cy = self.center_im.to_f64();
        [cx - self.half_range_x, cx + self.half_range_x,
         cy - self.half_range_y, cy + self.half_range_y]
    }

    /// Pixel step (complex units per pixel) for X axis.
    pub fn pixel_step_x(&self, display_w: u32) -> f64 {
        (2.0 * self.half_range_x) / (display_w as f64 - 1.0).max(1.0)
    }

    /// Pixel step (complex units per pixel) for Y axis.
    pub fn pixel_step_y(&self, display_h: u32) -> f64 {
        (2.0 * self.half_range_y) / (display_h as f64 - 1.0).max(1.0)
    }

    /// Ensure rug center precision is sufficient for current zoom depth.
    pub fn ensure_precision(&mut self) {
        let prec = self.required_precision();
        if self.center_re.prec() < prec {
            self.center_re = Float::with_val(prec, &self.center_re);
            self.center_im = Float::with_val(prec, &self.center_im);
        }
    }

    /// Required rug precision in bits for current zoom depth.
    fn required_precision(&self) -> u32 {
        let zoom_digits = if self.half_range_x > 0.0 {
            (-self.half_range_x.log10()).max(16.0) as u32
        } else {
            64
        };
        (zoom_digits * 4 + 64).max(128)
    }

    /// Reset center and range from default bounds for current fractal type.
    pub fn set_from_default_bounds(&mut self) {
        let b = self.fractal_type.default_bounds();
        self.center_re = Float::with_val(128, (b[0] + b[1]) / 2.0);
        self.center_im = Float::with_val(128, (b[2] + b[3]) / 2.0);
        self.half_range_x = (b[1] - b[0]) / 2.0;
        self.half_range_y = (b[3] - b[2]) / 2.0;
    }

    /// Set center and range from explicit bounds (for CLI --bounds).
    pub fn set_from_bounds(&mut self, bounds: [f64; 4]) {
        self.center_re = Float::with_val(128, (bounds[0] + bounds[1]) / 2.0);
        self.center_im = Float::with_val(128, (bounds[2] + bounds[3]) / 2.0);
        self.half_range_x = (bounds[1] - bounds[0]) / 2.0;
        self.half_range_y = (bounds[3] - bounds[2]) / 2.0;
    }

    /// Compute the nth roots of unity for z^n - 1 (Newton/Nova coloring).
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
    /// Uses rug center for Dekker splitting (f32 hi + f32 lo).
    pub fn to_gpu_params(&self, display_w: u32, display_h: u32, stride: u32) -> GpuParams {
        let cx_f64 = self.center_re.to_f64();
        let cy_f64 = self.center_im.to_f64();
        let step_x = self.pixel_step_x(display_w);
        let step_y = self.pixel_step_y(display_h);

        // Split f64 center into f32 hi + f32 lo (Dekker splitting)
        let cx_hi = cx_f64 as f32;
        let cx_lo = (cx_f64 - cx_hi as f64) as f32;
        let cy_hi = cy_f64 as f32;
        let cy_lo = (cy_f64 - cy_hi as f64) as f32;

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
            palette: self.palette.shader_index(),
            sample_index: 0,
            num_samples: 1,
            _pad: 0,
        }
    }
}

/// Pre-compute sub-pixel sample positions and weights for anti-aliasing.
/// Returns (offset_x, offset_y, weight) tuples in pixel units.
///
/// Uses jittered stratified sampling within [-0.5, +0.5]: each grid cell
/// gets a random offset instead of using the cell center. This breaks
/// moire patterns and converts structured aliasing into less visible noise.
/// Equal weights (box filter). Jitter is deterministic (seeded hash).
///
/// Quality levels:
///   ss=1: Off (1 sample at center)
///   ss=2: 6x6 grid within pixel — 36 samples
///   ss=3: 8x8 grid within pixel — 64 samples
pub fn compute_samples(ss: u32) -> Vec<(f32, f32, f32)> {
    if ss <= 1 {
        return vec![(0.0, 0.0, 1.0)];
    }

    let grid_n: u32 = match ss {
        2 => 6,
        _ => 8,
    };

    let mut samples = Vec::with_capacity((grid_n * grid_n) as usize);
    for sy in 0..grid_n {
        for sx in 0..grid_n {
            // Deterministic jitter via simple hash
            let seed = sx * 7919 + sy * 104729 + 31;
            let jx = hash_to_float(seed);
            let jy = hash_to_float(seed ^ 0x9E3779B9);
            let offset_x = -0.5 + (sx as f32 + jx) / grid_n as f32;
            let offset_y = -0.5 + (sy as f32 + jy) / grid_n as f32;
            samples.push((offset_x, offset_y, 1.0));
        }
    }
    samples
}

/// Simple deterministic hash → float in [0, 1).
fn hash_to_float(mut x: u32) -> f32 {
    x = x.wrapping_mul(0x45D9F3B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x45D9F3B);
    x ^= x >> 16;
    (x & 0x00FF_FFFF) as f32 / 16777216.0
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
    /// Reference orbit Z_n as double-single f32 quads: [re_hi, im_hi, re_lo, im_lo].
    /// Double-single gives ~48-bit mantissa (~15 decimal digits) vs f32's ~7 digits,
    /// pushing the perturbation precision artifact much deeper.
    pub orbit: Vec<[f32; 4]>,
    /// How many iterations before reference escaped (or max_iter).
    pub orbit_len: u32,
}

/// Compute a reference orbit at arbitrary precision using rug (GMP/MPFR).
/// Uses in-place operations to avoid heap allocations in the hot loop.
pub fn compute_reference_orbit(
    center_re: &Float,
    center_im: &Float,
    max_iter: u32,
    pixel_step: f64,
) -> PerturbData {
    // Precision: ~3.32 bits per decimal digit of zoom depth, plus safety margin
    let zoom_digits = if pixel_step > 0.0 {
        (-pixel_step.log10()).max(16.0) as u32
    } else {
        64
    };
    let precision = (zoom_digits * 4 + 64).max(128).max(center_re.prec());

    let c_re = Float::with_val(precision, center_re);
    let c_im = Float::with_val(precision, center_im);
    let mut z_re = Float::with_val(precision, 0.0);
    let mut z_im = Float::with_val(precision, 0.0);

    // Scratch variables reused every iteration (no allocations in hot loop)
    let mut zr2 = Float::new(precision);
    let mut zi2 = Float::new(precision);
    let mut zri = Float::new(precision);

    let mut orbit = Vec::with_capacity(max_iter as usize + 1);
    let escape_r2 = 256.0_f64;

    for _ in 0..max_iter {
        // Store orbit as double-single (Dekker split of f64 → f32 hi + f32 lo)
        let zr_f64 = z_re.to_f64();
        let zi_f64 = z_im.to_f64();
        let zr_hi = zr_f64 as f32;
        let zr_lo = (zr_f64 - zr_hi as f64) as f32;
        let zi_hi = zi_f64 as f32;
        let zi_lo = (zi_f64 - zi_hi as f64) as f32;
        orbit.push([zr_hi, zi_hi, zr_lo, zi_lo]);

        // z = z^2 + c using in-place ops (no heap allocs)
        zr2.assign(z_re.square_ref());       // zr2 = z_re^2
        zi2.assign(z_im.square_ref());       // zi2 = z_im^2
        zri.assign(&z_re * &z_im);          // zri = z_re * z_im

        z_re.assign(&zr2 - &zi2);           // z_re = zr2 - zi2
        z_re += &c_re;                       // z_re += c_re

        z_im.assign(&zri << 1u32);           // z_im = 2 * zri
        z_im += &c_im;                       // z_im += c_im

        // Escape check using f64 (avoids 2 arb-prec multiplications)
        let zr = z_re.to_f64();
        let zi = z_im.to_f64();
        if zr * zr + zi * zi > escape_r2 {
            let zr_hi = zr as f32;
            let zr_lo = (zr - zr_hi as f64) as f32;
            let zi_hi = zi as f32;
            let zi_lo = (zi - zi_hi as f64) as f32;
            orbit.push([zr_hi, zi_hi, zr_lo, zi_lo]);
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
    pub palette: u32,              // 4 bytes  (offset 80)
    pub sample_index: u32,         // 4 bytes  (offset 84) — which sub-pixel sample (0..N-1)
    pub num_samples: u32,          // 4 bytes  (offset 88) — total number of samples
    pub _pad: u32,                 // 4 bytes  (offset 92) — align to 96
}
// Total: 96 bytes
