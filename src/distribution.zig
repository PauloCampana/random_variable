//! Density/mass, probability, quantile and random number generation
//! functions for common probability distributions
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe modes
//! such as ±NaN, ±Inf, probabilities outside of the [0,1] interval and
//! certain parameters with value zero or negative.
//!
//! Random variable generation has 3 flavours: `single`, `buffer` and `alloc`,
//! `single` returns one generated number, `buffer` fills a slice with generated numbers,
//! `alloc` takes an allocator and the quantity to be generated then returns a heap allocated
//! slice, result must be freed by the caller.

pub const bernoulli = @import("distribution/bernoulli.zig");
pub const geometric = @import("distribution/geometric.zig");
pub const poisson = @import("distribution/poisson.zig");
pub const binomial = @import("distribution/binomial.zig");
pub const negativeBinomial = @import("distribution/negativeBinomial.zig");
pub const uniform = @import("distribution/uniform.zig");
pub const exponential = @import("distribution/exponential.zig");
pub const weibull = @import("distribution/weibull.zig");
pub const cauchy = @import("distribution/cauchy.zig");
pub const logistic = @import("distribution/logistic.zig");
pub const gamma = @import("distribution/gamma.zig");
pub const chiSquared = @import("distribution/chiSquared.zig");
pub const f = @import("distribution/f.zig");
pub const beta = @import("distribution/beta.zig");
pub const normal = @import("distribution/normal.zig");
pub const logNormal = @import("distribution/logNormal.zig");
pub const t = @import("distribution/t.zig");
