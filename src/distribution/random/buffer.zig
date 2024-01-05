//! Fills a buffer with random variables.
//!
//! First and second arguments are always
//! the buffer to be written and the rng engine,
//! the rest are the distribution's parameters.

const std = @import("std");
const implementation = @import("implementation.zig");

const Random = std.rand.Random;
const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan

/// prob ∈ [0,1]
pub fn bernoulli(buf: []f64, random: Random, prob: f64) []f64 {
    assert(0 <= prob and prob <= 1);
    for (buf) |*x| {
        x.* = implementation.bernoulli(random, prob);
    }
    return buf;
}

/// prob ∈ (0,1]
pub fn geometric(buf: []f64, random: Random, prob: f64) []f64 {
    assert(0 < prob and prob <= 1);
    for (buf) |*x| {
        x.* = implementation.geometric(random, prob);
    }
    return buf;
}

/// lambda ∈ (0,∞)
pub fn poisson(buf: []f64, random: Random, lambda: f64) []f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    for (buf) |*x| {
        x.* = implementation.poisson(random, lambda);
    }
    return buf;
}

/// size ∈ {0,1,2,⋯}, prob ∈ [0,1]
pub fn binomial(buf: []f64, random: Random, size: u64, prob: f64) []f64 {
    assert(0 <= prob and prob <= 1);
    for (buf) |*x| {
        x.* =  implementation.binomial(random, size, prob);
    }
    return buf;
}

/// size ∈ {1,2,3,⋯}, prob ∈ (0,1]
pub fn negativeBinomial(buf: []f64, random: Random, size: u64, prob: f64) []f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    for (buf) |*x| {
        x.* = implementation.negativeBinomial(random, size, prob);
    }
    return buf;
}

/// min and max ∈ (-∞,∞)
pub fn uniform(buf: []f64, random: Random, min: f64, max: f64) []f64 {
    assert(isFinite(min) and isFinite(max));
    for (buf) |*x| {
        x.* = implementation.uniform(random, min, max);
    }
    return buf;
}

/// rate ∈ (0,∞)
pub fn exponential(buf: []f64, random: Random, rate: f64) []f64 {
    assert(isFinite(rate));
    assert(rate > 0);
    for (buf) |*x| {
        x.* = implementation.exponential(random, rate);
    }
    return buf;
}

/// shape and rate ∈ (0,∞)
pub fn weibull(buf: []f64, random: Random, shape: f64, rate: f64) []f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    for (buf) |*x| {
        x.* = implementation.weibull(random, shape, rate);
    }
    return buf;
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn cauchy(buf: []f64, random: Random, location: f64, scale: f64) []f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    for (buf) |*x| {
        x.* = implementation.cauchy(random, location, scale);
    }
    return buf;
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn logistic(buf: []f64, random: Random, location: f64, scale: f64) []f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    for (buf) |*x| {
        x.* = implementation.logistic(random, location, scale);
    }
    return buf;
}

/// shape and rate ∈ (0,∞)
pub fn gamma(buf: []f64, random: Random, shape: f64, rate: f64) []f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    for (buf) |*x| {
        x.* = implementation.gamma(random, shape, rate);
    }
    return buf;
}

/// df ∈ (0,∞)
pub fn chiSquared(buf: []f64, random: Random, df: f64) []f64 {
    assert(isFinite(df));
    assert(df > 0);
    for (buf) |*x| {
        x.* = implementation.chiSquared(random, df);
    }
    return buf;
}

/// df1 and df2 ∈ (0,∞)
pub fn f(buf: []f64, random: Random, df1: f64, df2: f64) []f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    for (buf) |*x| {
        x.* = implementation.f(random, df1, df2);
    }
    return buf;
}

/// shape1 and shape2 ∈ (0,∞)
pub fn beta(buf: []f64, random: Random, shape1: f64, shape2: f64) []f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    for (buf) |*x| {
        x.* = implementation.beta(random, shape1, shape2);
    }
    return buf;
}

/// mean ∈ (-∞,∞), sd ∈ (0,∞)
pub fn normal(buf: []f64, random: Random, mean: f64, sd: f64) []f64 {
    assert(isFinite(mean) and isFinite(sd));
    assert(sd > 0);
    for (buf) |*x| {
        x.* = implementation.normal(random, mean, sd);
    }
    return buf;
}

/// meanlog ∈ (-∞,∞), sdlog ∈ (0,∞)
pub fn logNormal(buf: []f64, random: Random, meanlog: f64, sdlog: f64) []f64 {
    assert(isFinite(meanlog) and isFinite(sdlog));
    assert(sdlog > 0);
    for (buf) |*x| {
        x.* = implementation.logNormal(random, meanlog, sdlog);
    }
    return buf;
}

/// df ∈ (0,∞)
pub fn t(buf: []f64, random: Random, df: f64) []f64 {
    assert(isFinite(df));
    assert(df > 0);
    for (buf) |*x| {
        x.* = implementation.t(random, df);
    }
    return buf;
}
