//! Density/mass, probability, quantile and random number generation
//! functions for common probability distributions
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe modes
//! such as ±NaN, ±Inf, probabilities outside of the [0,1] interval and
//! certain parameters with value zero or negative.

pub const density = @import("distribution/density.zig");
pub const probability = @import("distribution/probability.zig");
pub const quantile = @import("distribution/quantile.zig");
pub const random = @import("distribution/random.zig");

test density {
    _ = density.normal(3, 0, 1);
    _ = density.gamma(10, 3, 5);
    _ = density.binomial(5, 10, 0.2);
}

test probability {
    _ = probability.normal(3, 0, 1);
    _ = probability.gamma(10, 3, 5);
    _ = probability.binomial(5, 10, 0.2);
}

test quantile {
    _ = quantile.normal(0.95, 0, 1);
    _ = quantile.gamma(0.95, 3, 5);
    _ = quantile.binomial(0.95, 10, 0.2);
}
