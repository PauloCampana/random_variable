//! Generates only a single random variable.
//!
//! First argument is always the rng engine,
//! the rest are the distribution's parameters.

const std = @import("std");
const implementation = @import("implementation.zig");

const Random = std.rand.Random;
const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan

/// prob ∈ [0,1]
pub fn bernoulli(random: Random, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    return implementation.bernoulli(random, prob);
}

/// prob ∈ (0,1]
pub fn geometric(random: Random, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    return implementation.geometric(random, prob);
}

/// lambda ∈ (0,∞)
pub fn poisson(random: Random, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    return implementation.poisson(random, lambda);
}

/// size ∈ {0,1,2,⋯}, prob ∈ [0,1]
pub fn binomial(random: Random, size: u64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    return implementation.binomial(random, size, prob);
}

/// size ∈ {1,2,3,⋯}, prob ∈ (0,1]
pub fn negativeBinomial(random: Random, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    return implementation.negativeBinomial(random, size, prob);
}

/// min and max ∈ (-∞,∞)
pub fn uniform(random: Random, min: f64, max: f64) f64 {
    assert(isFinite(min) and isFinite(max));
    return implementation.uniform(random, min, max);
}

/// rate ∈ (0,∞)
pub fn exponential(random: Random, rate: f64) f64 {
    assert(isFinite(rate));
    assert(rate > 0);
    return implementation.exponential(random, rate);
}

/// shape and rate ∈ (0,∞)
pub fn weibull(random: Random, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    return implementation.weibull(random, shape, rate);
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn cauchy(random: Random, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    return implementation.cauchy(random, location, scale);
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn logistic(random: Random, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    return implementation.logistic(random, location, scale);
}

/// shape and rate ∈ (0,∞)
pub fn gamma(random: Random, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    return implementation.gamma(random, shape, rate);
}

/// df ∈ (0,∞)
pub fn chiSquared(random: Random, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    return implementation.chiSquared(random, df);
}

/// df1 and df2 ∈ (0,∞)
pub fn f(random: Random, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    return implementation.f(random, df1, df2);
}

/// shape1 and shape2 ∈ (0,∞)
pub fn beta(random: Random, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    return implementation.beta(random, shape1, shape2);
}

/// mean ∈ (-∞,∞), sd ∈ (0,∞)
pub fn normal(random: Random, mean: f64, sd: f64) f64 {
    assert(isFinite(mean) and isFinite(sd));
    assert(sd > 0);
    return implementation.normal(random, mean, sd);
}

/// meanlog ∈ (-∞,∞), sdlog ∈ (0,∞)
pub fn logNormal(random: Random, meanlog: f64, sdlog: f64) f64 {
    assert(isFinite(meanlog) and isFinite(sdlog));
    assert(sdlog > 0);
    return implementation.logNormal(random, meanlog, sdlog);
}

/// df ∈ (0,∞)
pub fn t(random: Random, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    return implementation.t(random, df);
}
