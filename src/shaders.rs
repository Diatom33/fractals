//! WGSL source assembly. colorize.wgsl and median_finalize.wgsl share the
//! palette library (palettes.wgsl), spliced in at the //INCLUDE:palettes
//! marker so palette code exists in exactly one place.

const PALETTES: &str = include_str!("shaders/palettes.wgsl");
const MARKER: &str = "//INCLUDE:palettes";

fn assemble(body: &str) -> String {
    debug_assert!(body.contains(MARKER), "shader is missing the palette include marker");
    body.replace(MARKER, PALETTES)
}

pub fn colorize() -> String {
    assemble(include_str!("shaders/colorize.wgsl"))
}

pub fn median_finalize() -> String {
    assemble(include_str!("shaders/median_finalize.wgsl"))
}
