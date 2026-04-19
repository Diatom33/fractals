/// Fractal type definitions, parameters, and defaults.

use rug::{Assign, Float};
use rug::ops::NegAssign;

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
    Nebulabrot,
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
        FractalType::Nebulabrot,
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
            FractalType::Nebulabrot => "Nebulabrot",
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
            FractalType::Nebulabrot => 11,
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
            FractalType::Nebulabrot => [-2.5, 1.0, -1.25, 1.25],
        }
    }

    /// Whether this type uses its own rendering pipeline (not escape/newton).
    pub fn is_nebulabrot(&self) -> bool {
        matches!(self, FractalType::Nebulabrot)
    }

    /// Which controls should be visible for this fractal type.
    pub fn visible_controls(&self) -> Controls {
        match self {
            FractalType::Mandelbrot | FractalType::BurningShip | FractalType::Tricorn | FractalType::Celtic | FractalType::Perpendicular | FractalType::Buffalo | FractalType::Nebulabrot => Controls {
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
    pub coloring_param: f32,  // palette-specific parameter (thin-film k, aurora freq, storm steepness)
    pub use_median: bool,     // true = median iteration SS, false = Oklab accumulation SS
    // Nebulabrot-specific params
    pub nebula_iter_r: u32,
    pub nebula_iter_g: u32,
    pub nebula_iter_b: u32,
    pub nebula_samples_m: f64,  // millions of samples
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPalette {
    Classic,     // 0: HSV oscillating hue (original)
    Oklab,       // 1: Perceptually uniform lightness, varying hue
    Smooth,      // 2: iq-style cosine gradient
    Monochrome,  // 3: Single hue (blue), varying lightness
    ThinFilm,    // 4: Soap bubble / oil slick iridescence
    Aurora,      // 5: Midnight aurora green-violet bands
    Storm,       // 6: Oppressive brass murk with lightning
    Canopy,           // 7: Canopy with bright bokeh highlights (tone-mapped)
    Bioluminescence,  // 8: Deep-sea abyssal bioluminescence with depth-aware glow
    Steve,            // 9: STEVE atmospheric ribbon — pastel mauve with green picket fence
    InvertedPair,     // 10: High-contrast sinusoidal bands between complementary color pairs
}

impl ColorPalette {
    pub const ALL: &[ColorPalette] = &[
        ColorPalette::Classic,
        ColorPalette::Oklab,
        ColorPalette::Smooth,
        ColorPalette::Monochrome,
        ColorPalette::ThinFilm,
        ColorPalette::Aurora,
        ColorPalette::Storm,
        ColorPalette::Canopy,
        ColorPalette::Bioluminescence,
        ColorPalette::Steve,
        ColorPalette::InvertedPair,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ColorPalette::Classic => "Classic HSV",
            ColorPalette::Oklab => "Oklab Uniform",
            ColorPalette::Smooth => "Smooth Gradient",
            ColorPalette::Monochrome => "Monochrome",
            ColorPalette::ThinFilm => "Thin Film",
            ColorPalette::Aurora => "Midnight Aurora",
            ColorPalette::Storm => "Storm",
            ColorPalette::Canopy => "Canopy",
            ColorPalette::Bioluminescence => "Bioluminescence",
            ColorPalette::Steve => "STEVE",
            ColorPalette::InvertedPair => "Inverted Pair",
        }
    }

    pub fn shader_index(&self) -> u32 {
        match self {
            ColorPalette::Classic => 0,
            ColorPalette::Oklab => 1,
            ColorPalette::Smooth => 2,
            ColorPalette::Monochrome => 3,
            ColorPalette::ThinFilm => 4,
            ColorPalette::Aurora => 5,
            ColorPalette::Storm => 6,
            ColorPalette::Canopy => 7,
            ColorPalette::Bioluminescence => 8,
            ColorPalette::Steve => 9,
            ColorPalette::InvertedPair => 10,
        }
    }

    /// Default coloring_param value for this palette.
    pub fn default_param(&self) -> f32 {
        match self {
            ColorPalette::ThinFilm => 2.0,   // angular lobe count
            ColorPalette::Aurora => 3.0,      // band frequency
            ColorPalette::Storm => 1.0,       // noise scale
            ColorPalette::Canopy => 3.0,      // trap scale
            ColorPalette::Bioluminescence => 5.0, // murkiness
            ColorPalette::InvertedPair => 0.08, // band frequency
            ColorPalette::Steve => 18.0,       // picket-fence post density
            _ => 0.0,
        }
    }

    /// Whether this palette reads neighbor pixels (screen-space gradients/glow).
    /// These palettes can't be multi-sample averaged — gradient features shift with
    /// sub-pixel offsets, producing noise instead of smooth highlights.
    pub fn uses_neighbor_sampling(&self) -> bool {
        matches!(self, ColorPalette::Storm | ColorPalette::Bioluminescence | ColorPalette::Steve)
    }

    /// Whether this palette uses the coloring_param slider.
    pub fn has_param(&self) -> bool {
        matches!(self, ColorPalette::ThinFilm | ColorPalette::Aurora | ColorPalette::Storm | ColorPalette::Canopy | ColorPalette::Bioluminescence | ColorPalette::InvertedPair | ColorPalette::Steve)
    }

    /// Label for the coloring_param slider.
    pub fn param_label(&self) -> &'static str {
        match self {
            ColorPalette::ThinFilm => "Iridescence lobes",
            ColorPalette::Aurora => "Band frequency",
            ColorPalette::Storm => "Noise scale",
            ColorPalette::Canopy => "Trap scale",
            ColorPalette::Bioluminescence => "Murkiness",
            ColorPalette::InvertedPair => "Band frequency",
            ColorPalette::Steve => "Post density",
            _ => "",
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
            coloring_param: ColorPalette::Oklab.default_param(),
            use_median: true,
            nebula_iter_r: 5000,
            nebula_iter_g: 500,
            nebula_iter_b: 50,
            nebula_samples_m: 10.0,
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

        // Compute noise seed: center in noise-cell units (fractional part).
        // Using rug for arbitrary precision so panning is perfectly smooth at any zoom.
        // noise_cell_size = 55 pixels * coloring_param * pixel_step (complex-plane units).
        let noise_cell_px = 55.0_f64 * (self.coloring_param as f64).max(0.1);
        let cell_size_x = noise_cell_px * step_x;
        let cell_size_y = noise_cell_px * step_y;
        let noise_seed_x = if cell_size_x > 0.0 {
            let cell = Float::with_val(128, cell_size_x);
            let ratio = Float::with_val(128, &self.center_re / cell);
            let floor_val = ratio.clone().floor();
            Float::with_val(128, ratio - floor_val).to_f64() as f32
        } else {
            0.0f32
        };
        let noise_seed_y = if cell_size_y > 0.0 {
            let cell = Float::with_val(128, cell_size_y);
            let ratio = Float::with_val(128, &self.center_im / cell);
            let floor_val = ratio.clone().floor();
            Float::with_val(128, ratio - floor_val).to_f64() as f32
        } else {
            0.0f32
        };

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
            coloring_param: self.coloring_param,
            real_pixel_step: [step_x as f32, step_y as f32],
            noise_seed: [noise_seed_x, noise_seed_y],
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
    pub bla_num_levels: u32,    // 0 = BLA disabled
    pub _pad: u32,
}

/// One BLA node: applies 2^level perturbation iterations as δ ← A·δ + B·δc.
/// A and B mantissas use double-single (hi, lo) for ~14-digit precision matching
/// the DS-tracked δ — without this, multiplying DS δ by single-f32 A would silently
/// truncate the lo bits of δ on every BLA step.
/// Layout matches WGSL struct (48 bytes, 16-byte aligned: 4 vec4s).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlaCoeff {
    pub a: [f32; 4],         // 16 bytes — A as DS complex (re_hi, re_lo, im_hi, im_lo)
    pub b: [f32; 4],         // 16 bytes — B as DS complex
    pub a_exp: i32,          // 4 bytes
    pub b_exp: i32,          // 4 bytes
    pub radius_log2: f32,    // 4 bytes — node valid when log2(|δ|) ≤ radius_log2
    pub _pad: u32,           // 4 bytes (pad to 48 bytes)
}

impl Default for BlaCoeff {
    fn default() -> Self {
        // Default = invalid node (very negative radius_log2 → never selected)
        Self {
            a: [0.0; 4], b: [0.0; 4],
            a_exp: 0, b_exp: 0,
            radius_log2: -1.0e30,
            _pad: 0,
        }
    }
}

/// Split f64 into double-single f32 (hi + lo). |lo| ≤ ulp(hi)/2.
#[inline]
fn split_f64_to_ds(x: f64) -> (f32, f32) {
    let hi = x as f32;
    let lo = (x - hi as f64) as f32;
    (hi, lo)
}

/// Data for perturbation rendering: reference orbit + metadata + optional BLA tree.
pub struct PerturbData {
    /// Reference orbit Z_n as double-single f32 quads: [re_hi, im_hi, re_lo, im_lo].
    /// Double-single gives ~48-bit mantissa (~15 decimal digits) vs f32's ~7 digits,
    /// pushing the perturbation precision artifact much deeper.
    pub orbit: Vec<[f32; 4]>,
    /// How many iterations before reference escaped (or max_iter).
    pub orbit_len: u32,
    /// Optional BLA tree (Mandelbrot only). Layout: [level][n], padded so each
    /// level has ref_len entries. Entry at (level, n) covers iterations [n, n+2^level).
    /// Empty if BLA wasn't built.
    pub bla: Vec<BlaCoeff>,
    /// Number of BLA levels. 0 = BLA not built.
    pub bla_num_levels: u32,
}

/// Extended-range complex number: (re + i·im) · 2^exp. Used for BLA construction
/// where coefficients can grow/shrink exponentially with iteration count.
/// Mantissas held in f64 for combine accuracy; final storage truncates to f32.
#[derive(Debug, Clone, Copy)]
struct ExtComplex {
    re: f64,
    im: f64,
    exp: i32,
}

impl ExtComplex {
    const ZERO: Self = Self { re: 0.0, im: 0.0, exp: 0 };

    fn from_rug(re: &Float, im: &Float) -> Self {
        let mut c = Self {
            re: re.to_f64(),
            im: im.to_f64(),
            exp: 0,
        };
        c.renormalize();
        c
    }

    /// Re-anchor mantissa so max(|re|, |im|) is in [1, 2).
    fn renormalize(&mut self) {
        let mag = self.re.abs().max(self.im.abs());
        if mag > 0.0 && mag.is_finite() {
            let shift = mag.log2().floor() as i32;
            let scale = (-shift as f64).exp2();
            self.re *= scale;
            self.im *= scale;
            self.exp += shift;
        } else {
            *self = Self::ZERO;
        }
    }

    fn mul(self, other: Self) -> Self {
        let mut r = Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
            exp: self.exp + other.exp,
        };
        r.renormalize();
        r
    }

    fn add(self, other: Self) -> Self {
        if self.re == 0.0 && self.im == 0.0 { return other; }
        if other.re == 0.0 && other.im == 0.0 { return self; }
        let exp = self.exp.max(other.exp);
        let s_scale = (self.exp - exp) as f64;
        let o_scale = (other.exp - exp) as f64;
        let s_factor = s_scale.exp2();
        let o_factor = o_scale.exp2();
        let mut r = Self {
            re: self.re * s_factor + other.re * o_factor,
            im: self.im * s_factor + other.im * o_factor,
            exp,
        };
        r.renormalize();
        r
    }

    /// log2 of magnitude: log2(|self|) = exp + log2(sqrt(re²+im²)).
    /// Returns very negative for zero.
    fn log2_mag(self) -> f64 {
        let m2 = self.re * self.re + self.im * self.im;
        if m2 > 0.0 {
            self.exp as f64 + 0.5 * m2.log2()
        } else {
            -1.0e30
        }
    }

    fn to_coeff_a(self, b: Self, radius_log2: f32) -> BlaCoeff {
        let (a_re_hi, a_re_lo) = split_f64_to_ds(self.re);
        let (a_im_hi, a_im_lo) = split_f64_to_ds(self.im);
        let (b_re_hi, b_re_lo) = split_f64_to_ds(b.re);
        let (b_im_hi, b_im_lo) = split_f64_to_ds(b.im);
        BlaCoeff {
            a: [a_re_hi, a_re_lo, a_im_hi, a_im_lo],
            b: [b_re_hi, b_re_lo, b_im_hi, b_im_lo],
            a_exp: self.exp,
            b_exp: b.exp,
            radius_log2,
            _pad: 0,
        }
    }
}


/// Compute a reference orbit for any z²+c escape-time variant.
/// Nebulabrot sampling GPU params. Must match NebulaParams in nebula_sample.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NebulaGpuParams {
    pub resolution: [u32; 2],
    pub stride: u32,
    pub max_iter_r: u32,
    pub max_iter_g: u32,
    pub max_iter_b: u32,
    pub samples_per_thread: u32,
    pub dispatch_index: u32,
    pub sample_min: [f32; 2],
    pub sample_max: [f32; 2],
    pub view_min: [f32; 2],
    pub view_max: [f32; 2],
}

/// Nebulabrot finalize GPU params. Must match NebFinParams in nebula_finalize.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NebulaFinParams {
    pub resolution: [u32; 2],
    pub stride: u32,
    pub max_r: u32,
    pub max_g: u32,
    pub max_b: u32,
    pub _pad: [u32; 2],
}

/// The variant determines how z is modified before/after squaring.
/// Stores the raw Z_n (NOT the modified version) — the GPU computes
/// signs and abs from the stored values as needed.
pub fn compute_variant_reference_orbit(
    center_re: &Float,
    center_im: &Float,
    max_iter: u32,
    pixel_step: f64,
    fractal_type: FractalType,
    julia_c: Option<(f64, f64)>,
) -> PerturbData {
    let zoom_digits = if pixel_step > 0.0 {
        (-pixel_step.log10()).max(16.0) as u32
    } else {
        64
    };
    let precision = (zoom_digits * 4 + 64).max(128).max(center_re.prec());

    let is_julia = fractal_type == FractalType::Julia;
    let (c_re, c_im, mut z_re, mut z_im) = if is_julia {
        let jc = julia_c.unwrap_or((0.0, 0.0));
        (
            Float::with_val(precision, jc.0),
            Float::with_val(precision, jc.1),
            Float::with_val(precision, center_re),
            Float::with_val(precision, center_im),
        )
    } else {
        (
            Float::with_val(precision, center_re),
            Float::with_val(precision, center_im),
            Float::with_val(precision, 0.0),
            Float::with_val(precision, 0.0),
        )
    };

    let mut zr2 = Float::new(precision);
    let mut zi2 = Float::new(precision);
    let mut zri = Float::new(precision);

    let mut orbit = Vec::with_capacity(max_iter as usize + 1);
    let escape_r2 = 256.0_f64;

    for _ in 0..max_iter {
        let zr_f64 = z_re.to_f64();
        let zi_f64 = z_im.to_f64();
        let zr_hi = zr_f64 as f32;
        let zr_lo = (zr_f64 - zr_hi as f64) as f32;
        let zi_hi = zi_f64 as f32;
        let zi_lo = (zi_f64 - zi_hi as f64) as f32;
        orbit.push([zr_hi, zi_hi, zr_lo, zi_lo]);

        // Apply variant-specific iteration
        match fractal_type {
            FractalType::BurningShip => {
                // z = (|Re(z)| + i|Im(z)|)² + c
                z_re.abs_mut();
                z_im.abs_mut();
                zr2.assign(z_re.square_ref());
                zi2.assign(z_im.square_ref());
                zri.assign(&z_re * &z_im);
                z_re.assign(&zr2 - &zi2);
                z_re += &c_re;
                z_im.assign(&zri << 1u32);
                z_im += &c_im;
            }
            FractalType::Tricorn => {
                // z = conj(z)² + c = (Re(z)² - Im(z)², -2·Re(z)·Im(z)) + c
                zr2.assign(z_re.square_ref());
                zi2.assign(z_im.square_ref());
                zri.assign(&z_re * &z_im);
                z_re.assign(&zr2 - &zi2);
                z_re += &c_re;
                z_im.assign(&zri << 1u32);
                z_im.neg_assign(); // negate: conjugation
                z_im += &c_im;
            }
            FractalType::Celtic => {
                // z² then |Re(z²)|, Im(z²) unchanged
                zr2.assign(z_re.square_ref());
                zi2.assign(z_im.square_ref());
                zri.assign(&z_re * &z_im);
                z_re.assign(&zr2 - &zi2);
                z_re.abs_mut(); // Celtic: |Re(z²)|
                z_re += &c_re;
                z_im.assign(&zri << 1u32);
                z_im += &c_im;
            }
            FractalType::Perpendicular => {
                // Re: standard z². Im: -2·|Re(z)|·Im(z)
                let sign_re = z_re.to_f64() >= 0.0;
                zr2.assign(z_re.square_ref());
                zi2.assign(z_im.square_ref());
                // Im = -2·|Re(z)|·Im(z) = -sign(Re(z))·2·Re(z)·Im(z)
                zri.assign(&z_re * &z_im);
                z_re.assign(&zr2 - &zi2);
                z_re += &c_re;
                z_im.assign(&zri << 1u32);
                if sign_re {
                    z_im.neg_assign();
                }
                z_im += &c_im;
            }
            FractalType::Buffalo => {
                // |Re(z²)|, -|Im(z²)|
                zr2.assign(z_re.square_ref());
                zi2.assign(z_im.square_ref());
                zri.assign(&z_re * &z_im);
                z_re.assign(&zr2 - &zi2);
                z_re.abs_mut(); // |Re(z²)|
                z_re += &c_re;
                z_im.assign(&zri << 1u32);
                z_im.abs_mut();
                z_im.neg_assign(); // -|Im(z²)|
                z_im += &c_im;
            }
            _ => {
                // Standard Mandelbrot / Julia: z = z² + c
                zr2.assign(z_re.square_ref());
                zi2.assign(z_im.square_ref());
                zri.assign(&z_re * &z_im);
                z_re.assign(&zr2 - &zi2);
                z_re += &c_re;
                z_im.assign(&zri << 1u32);
                z_im += &c_im;
            }
        }

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
    PerturbData { orbit, orbit_len, bla: Vec::new(), bla_num_levels: 0 }
}

/// Mandelbrot-only reference orbit + BLA tree. The BLA tree lets the GPU skip
/// many iterations at once when |δ| stays inside a validity radius.
///
/// Tree layout: flat array indexed as `bla[n * num_levels + level]`. Entry at
/// (n, level) covers iterations [n, n + 2^level). Each level k entry merges
/// two level-(k-1) entries with offset 2^(k-1).
///
/// At level 0: A = 2·Z_n, B = 1, radius = eps · |2 Z_n|.
/// At level k+1: A = A2·A1, B = A2·B1 + B2, radius = min(r1, (r2 − |B1|·|δc_max|) / |A1|).
pub fn compute_mandelbrot_with_bla(
    center_re: &Float,
    center_im: &Float,
    max_iter: u32,
    pixel_step: f64,
    delta_c_max: f64,
    eps: f64,
) -> PerturbData {
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
    let mut zr2 = Float::new(precision);
    let mut zi2 = Float::new(precision);
    let mut zri = Float::new(precision);

    let mut orbit: Vec<[f32; 4]> = Vec::with_capacity(max_iter as usize + 1);
    // Cache 2·Z_n as ExtComplex for level-0 BLA construction
    let mut two_z: Vec<ExtComplex> = Vec::with_capacity(max_iter as usize + 1);
    let escape_r2 = 256.0_f64;

    for _ in 0..max_iter {
        let zr_f64 = z_re.to_f64();
        let zi_f64 = z_im.to_f64();
        let zr_hi = zr_f64 as f32;
        let zr_lo = (zr_f64 - zr_hi as f64) as f32;
        let zi_hi = zi_f64 as f32;
        let zi_lo = (zi_f64 - zi_hi as f64) as f32;
        orbit.push([zr_hi, zi_hi, zr_lo, zi_lo]);

        // Capture 2·Z_n at full rug precision before iterating
        let two_zr = Float::with_val(precision, &z_re * 2);
        let two_zi = Float::with_val(precision, &z_im * 2);
        two_z.push(ExtComplex::from_rug(&two_zr, &two_zi));

        // Standard Mandelbrot: z = z² + c
        zr2.assign(z_re.square_ref());
        zi2.assign(z_im.square_ref());
        zri.assign(&z_re * &z_im);
        z_re.assign(&zr2 - &zi2);
        z_re += &c_re;
        z_im.assign(&zri << 1u32);
        z_im += &c_im;

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
    let ref_len = two_z.len();

    // BLA tree: number of levels needed to span the orbit.
    let num_levels: u32 = (ref_len.max(2) as f64).log2().ceil() as u32 + 1;
    let total = ref_len * num_levels as usize;
    let mut bla = vec![BlaCoeff::default(); total];

    // Level 0: A = 2·Z_n, B = 1, radius = eps · |2 Z_n|.
    // When |2Z_n| is tiny, radius is tiny too — pixel won't use BLA there, falls back to single step.
    let b0 = ExtComplex { re: 1.0, im: 0.0, exp: 0 };
    for n in 0..ref_len {
        let a = two_z[n];
        let log2_a = a.log2_mag();
        // radius = eps · |A| → log2(radius) = log2(eps) + log2(|A|)
        let radius_log2 = if log2_a > -1.0e29 {
            (eps.log2() + log2_a) as f32
        } else {
            -1.0e30
        };
        bla[n * num_levels as usize] = a.to_coeff_a(b0, radius_log2);
    }

    // Higher levels: combine two level-(k-1) entries with stride 2^(k-1).
    let log_dc = if delta_c_max > 0.0 { delta_c_max.log2() } else { -1.0e30 };
    for level in 1..num_levels as usize {
        let stride = 1usize << (level - 1);
        for n in 0..ref_len {
            let n_mid = n + stride;
            if n_mid >= ref_len {
                // Can't combine — leave as default (invalid).
                continue;
            }
            let prev = level - 1;
            let bla1 = bla[n * num_levels as usize + prev];
            let bla2 = bla[n_mid * num_levels as usize + prev];

            // Skip if either half is invalid (shouldn't happen at level 1+ from valid level 0)
            if bla1.radius_log2 < -1.0e29 || bla2.radius_log2 < -1.0e29 {
                continue;
            }

            // Reconstruct DS coefficients (hi + lo) into f64 for combine accuracy.
            let a1 = ExtComplex {
                re: bla1.a[0] as f64 + bla1.a[1] as f64,
                im: bla1.a[2] as f64 + bla1.a[3] as f64,
                exp: bla1.a_exp,
            };
            let b1 = ExtComplex {
                re: bla1.b[0] as f64 + bla1.b[1] as f64,
                im: bla1.b[2] as f64 + bla1.b[3] as f64,
                exp: bla1.b_exp,
            };
            let a2 = ExtComplex {
                re: bla2.a[0] as f64 + bla2.a[1] as f64,
                im: bla2.a[2] as f64 + bla2.a[3] as f64,
                exp: bla2.a_exp,
            };
            let b2 = ExtComplex {
                re: bla2.b[0] as f64 + bla2.b[1] as f64,
                im: bla2.b[2] as f64 + bla2.b[3] as f64,
                exp: bla2.b_exp,
            };

            let a = a2.mul(a1);
            let b = a2.mul(b1).add(b2);

            // radius = min(r1, max(0, (r2 - |B1|·|δc_max|) / |A1|))
            let log_a1 = a1.log2_mag();
            let log_b1 = b1.log2_mag();
            let log_b1_dc = log_b1 + log_dc;
            let r2 = bla2.radius_log2 as f64;

            // Compute log2(2^r2 - 2^log_b1_dc) safely.
            let diff_log = if r2 > log_b1_dc {
                let diff = log_b1_dc - r2;
                let factor = 1.0 - diff.exp2();
                if factor > 0.0 { r2 + factor.log2() } else { -1.0e30 }
            } else {
                -1.0e30
            };

            let candidate = if diff_log > -1.0e29 {
                diff_log - log_a1
            } else {
                -1.0e30
            };
            let combined_radius = (bla1.radius_log2 as f64).min(candidate);

            bla[n * num_levels as usize + level] = a.to_coeff_a(b, combined_radius as f32);
        }
    }

    PerturbData {
        orbit,
        orbit_len,
        bla,
        bla_num_levels: num_levels,
    }
}

/// GPU-side uniform struct. Must match WGSL layout exactly.
/// Uses center + pixel_step instead of raw bounds
/// to enable double-single (emulated f64) precision in shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParams {
    pub center_hi: [f32; 2],       // 8 bytes  (offset 0)  — view center (high part)
    pub center_lo: [f32; 2],       // 8 bytes  (offset 8)  — view center (low part, sub-ULP)
    pub pixel_step: [f32; 2],      // 8 bytes  (offset 16) — complex units per pixel (mantissa in perturbation)
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
    pub coloring_param: f32,       // 4 bytes  (offset 92) — palette-specific parameter
    pub real_pixel_step: [f32; 2], // 8 bytes  (offset 96) — always true pixel step (even in perturbation)
    pub noise_seed: [f32; 2],     // 8 bytes  (offset 104) — fBm noise seed from rug center
}
// Total: 112 bytes
