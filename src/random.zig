//! Random variable generation
//! for common probability distributions
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe
//! such as ±NaN, ±Inf, probabilities outside [0,1],
//! negative or zero shape, df, rate or scale parameters.

pub const Single = @import("random/single.zig").Single;
pub const Buffer = @import("random/buffer.zig").Buffer;
pub const Alloc = @import("random/alloc.zig").Alloc;
