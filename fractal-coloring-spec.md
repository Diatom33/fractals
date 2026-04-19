# Fractal Coloring Schemes — Implementation Spec

## Context

We're building coloring/post-processing systems for a fractal renderer (Mandelbrot, Burning Ship, etc). The goal is to move beyond standard HSV rainbow escape-time coloring into something more visually striking and physically motivated. There are several independent schemes described below. They share some common orbit statistics but are otherwise separate rendering modes.

---

## Common Orbit Statistics to Extract

For each pixel's orbit `z_0, z_1, ..., z_n`, compute and store:

- **Smooth iteration count**: `n_smooth = n + 1 - log2(log(|z_final|))` — continuous version of escape count
- **Final angle**: `arg(z_final)` — breaks radial symmetry, useful as a secondary coloring axis
- **Orbit average radius**: `mean(|z_i|)` over the orbit
- **Triangle inequality average**: measures how "direct" the escape path was, gives smooth shading independent of iteration count
- **Curvature estimate**: `mean(arg(z_{i+1} - z_i) - arg(z_i - z_{i-1}))` — highlights filaments and spirals
- **Distance estimate** (requires computing derivative alongside orbit): `d = 2|z| · log|z| / |z'|` — essential for boundary glow effects
- **Gradient of smooth iteration count**: `|∇n_smooth|` computed via finite differences from the n_smooth image — identifies structural features (edges, filaments, spirals) vs smooth interiors

Having 2-3 independent statistics enables 2D/3D palette lookups, which is what breaks the "it's obviously a color ramp" look.

---

## Scheme 1: Thin-Film Interference (Soap Bubble / Oil Slick)

### Concept
Model colors as thin-film interference — different "thicknesses" produce different iridescent color sequences. Two parameters: optical thickness and viewing angle.

### Per-pixel math
Treat R, G, B as three representative wavelengths (~650nm, ~550nm, ~450nm). Compute:

```
R = sin²(π · t_eff / 0.650)
G = sin²(π · t_eff / 0.550)
B = sin²(π · t_eff / 0.450)
```

where:
```
t_base    = f(n_smooth)          # map smooth iteration count to "optical thickness" in µm
t_eff     = t_base / cos(arg(z_final) * k)   # viewing angle modulation
```

### Parameters
- The mapping `f()` for `t_base` controls the base color sequence — linear, sqrt, log all give different looks. Experiment.
- `k` controls angular lobe count: low (1-2) = big swirling patches, high = fine chaotic shimmer
- The multiplicative (not additive) angle modulation is important — it makes the color a 2D function, producing swirling oil-slick patches rather than contour lines

### Optional: head tracking
If webcam head tracking is available (e.g. MediaPipe face mesh), the physical head position relative to screen gives a real incidence angle. Use that instead of `arg(z_final)` for the angle parameter. Smooth the tracking signal with an EMA, ~100ms time constant.

---

## Scheme 2: Abyssal Bioluminescence

### Concept
Deep-sea underwater scene. Structural features (filaments, spirals, edges) are bioluminescent creatures emitting light. Smooth iteration-band interiors are dark water. Light scatters laterally through water with depth (iteration count) acting as a third spatial dimension — light attenuates across iteration-count differences.

### Step 1: Identify emitters
Emitter signal = `|∇n_smooth|` (spatial gradient magnitude of smooth iteration count). Where iteration count changes rapidly → bright filament. Where it's flat → dark water.

Optional secondary emitter: triangle inequality average at low intensity — ambient "marine snow" particulate glow.

### Step 2: Color the emitters
Base emitter color: cyan-green (hue ~170-180°). Modulate hue with curvature estimate — high curvature spirals get a different hue than gentle curves. Different "species."

### Step 3: Bloom with depth-dependent scattering
This is the core of the effect. The bloom convolution kernel between two pixels depends on BOTH spatial distance and iteration-count difference:

```
weight(r, Δn) = exp(-r / σ_spatial) · exp(-|Δn| / σ_depth)
```

- Light scatters freely between spatially nearby pixels at similar iteration counts
- Light is absorbed crossing iteration-count boundaries
- Two bright filaments that are spatially adjacent but at different iteration depths don't bleed into each other

The kernel shape should be exponential (not Gaussian) — fatter tails give the eerie long-range glow of real light scattering in murky water.

### Step 4: Wavelength-dependent scattering
Run separate bloom passes per channel with different kernel widths:
- Green channel: tighter kernel (green light penetrates water farther)
- Blue channel: wider kernel (blue scatters more)

Result: glow halos shift from green at center to blue at edges.

### Step 5: Beer-Lambert attenuation with adaptive depth
Apply `I_observed = I_emitted · exp(-α · |n_smooth - n_viewer|)` where:
- `α` controls murkiness (tunable parameter — high = claustrophobic deep trench, low = clearer water)
- `n_viewer` = median or mode of visible iteration counts in the current frame
- This means the coloring adapts automatically as you zoom — always reveals local structure, always suggests more hidden above and below

### Step 6: Wavelength-dependent depth attenuation
Red attenuates fastest with depth-distance, so structures far from `n_viewer` in iteration space lose warmth and shift toward pure blue glow. Consistent with real ocean physics.

### Implementation note
The depth-aware bloom is the expensive part. If doing it naively per-pixel it's O(n²) in pixel count times the kernel size in iteration space. Consider:
- Binning pixels by iteration count into layers, blurring each layer spatially, then compositing with cross-layer attenuation
- Separable approximations if possible
- Doing it at reduced resolution and upsampling the bloom layer

---

## Scheme 3: Storm Threshold

### Concept
Oppressive, desaturated, high-contrast storm scene. Most of the image is muddy and dark. Rare bright structures are "lightning."

### Step 1: Sigmoid contrast crush
Render a base value in [0,1] from orbit statistics, then apply a strong sigmoid:
```
v = 1 / (1 + exp(-k * (x - 0.5)))
```
with high `k`. This crushes midtones toward dark center, only extremes pop. Most pixels end up in oppressive murk.

### Step 2: Base palette
Work in HSL. Low saturation (0.1–0.3), mid-range value, warm-neutral hue (30–50°, tarnished brass/ash).

### Step 3: Lightning from distance estimate
The distance estimate near the Mandelbrot boundary produces infinitely fine branching structures. Threshold it low — pixels where the distance estimate is below threshold become "lightning bolts."

Override these pixels: near-white, slight blue-violet tint, sharp falloff. They already have correct branching morphology for free.

Target: ~2% of pixels bright, all in jagged filamentary shapes. The rest stays in brass-gray murk.

---

## Scheme 4: Primordial Canopy

### Concept
Two-layer compositing: dappled sunlight canopy over jewel-bright scattered creatures/flowers.

### Layer 1: Canopy light (low frequency)
Take smooth iteration count, apply a very broad Gaussian blur (or render at very coarse resolution and upsample). Map to a warm golden-green gradient. This is the slow-varying dappled sunlight — big soft shafts. Think of it as a lighting/illumination pass.

### Layer 2: Fractal detail with orbit traps
Place several "trap points" in the complex plane. For each orbit, track minimum distance to each trap point. When orbit passes close to a trap, that pixel gets a high-chroma accent color assigned to that trap.

- Different traps → different jewel tones (ruby, sapphire, amber, emerald)
- Trap distance controls accent intensity (closer = brighter)
- Result: scattered bright gems fading into the green background

### Compositing
Multiply Layer 1 over the final image. Even jewel-bright motes are dimmer in the "shadows" between sunlight shafts. This compositional structure is what makes it feel like a place rather than a texture.

---

## Scheme 5: Midnight Aurora

### Concept
Frozen lake at midnight with aurora overhead. Literal reflection in the image.

### Aurora palette
Dark everywhere except in narrow iteration bands where it flares through a green-violet gradient. Use `smoothstep` (not linear) for band transitions — soft-edged luminous ribbons, not hard bands.

### Reflection
Render the fractal, flip vertically across a horizontal center line, composite the flipped version underneath at:
- Reduced brightness
- Slight additional Gaussian blur (imperfect reflection in ice)
- Blue-shifted (reflected aurora should be dimmer and cooler)

Aurora colors appear only in the upper portion, fading toward the reflection line.

---

## Scheme 6: Nebulabrot Hybrid

### Concept
Buddhabrot/Nebulabrot rendering (orbit density accumulation) combined with the post-processing from the other schemes.

### Core rendering
Instead of coloring pixels by escape time, pick random starting points `c`, iterate `z → z² + c`, and for escaping orbits, trace the entire trajectory and increment a 2D hit-count histogram at every pixel visited. Three separate histograms with different iteration limits (e.g. R=2000, G=200, B=20) → assign to RGB channels. Apply sqrt or cbrt scaling to the histograms before normalization.

### Hybrid opportunity
The Nebulabrot density structures already have organic, filamentary quality. Apply the bioluminescence bloom post-processing (Scheme 2) on top of the Nebulabrot render — use the hit-count density as the emitter signal instead of `|∇n_smooth|`.

### Caveat
Nebulabrot rendering is fundamentally more expensive than escape-time and doesn't zoom well. Zoomed renders require Metropolis-Hastings importance sampling or similar. This is a batch/offline rendering mode, not real-time.

---

## Head Tracking Parallax (applies to any scheme)

If webcam head tracking is available, use lateral head position to create a parallax/holographic depth effect:

- Render iteration bands as conceptual depth layers
- Shift each layer laterally based on head position, scaled by iteration count
- Low iteration = near (barely moves), high iteration = far (moves more)
- This gives genuine perceived 3D depth — filigree near boundary floats behind large bulbs

Use MediaPipe face mesh in browser. Smooth with EMA ~100ms. Only lateral displacement is meaningful — forward/back head movement doesn't add anything that normal vision doesn't already provide.

---

## General Notes

- All schemes benefit from having multiple independent orbit statistics. Compute them all even if a given scheme only uses a subset — it's cheap relative to the iteration itself.
- The bioluminescence bloom with depth-aware kernel is the most novel and complex component. It could be its own reusable module.
- For real-time rendering, the post-processing passes (bloom, compositing) should be GPU shaders. The orbit statistics can be computed during the escape-time iteration pass and stored in a multi-channel framebuffer.
- Consider exposing the key parameters (murkiness α, angular lobe count k, sigmoid steepness, etc.) as interactive sliders for exploration.
