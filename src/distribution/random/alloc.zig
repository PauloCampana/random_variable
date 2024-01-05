//! Uses an allocator to provide a slice of random variables.
//!
//! First and second and third arguments are always
//! the allocator, the rng engine,
//! and the number of variables to be generated,
//! the rest are the distribution's parameters.

const std = @import("std");
const implementation = @import("implementation.zig");

const Allocator = std.mem.Allocator;
const Random = std.rand.Random;
const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan

/// prob ∈ [0,1]
pub fn bernoulli(allocator: Allocator, random: Random, n: usize, prob: f64) ![]f64 {
    assert(0 <= prob and prob <= 1);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.bernoulli(random, prob);
    }
    return slice;
}

/// prob ∈ (0,1]
pub fn geometric(allocator: Allocator, random: Random, n: usize, prob: f64) ![]f64 {
    assert(0 < prob and prob <= 1);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.geometric(random, prob);
    }
    return slice;
}

/// lambda ∈ (0,∞)
pub fn poisson(allocator: Allocator, random: Random, n: usize, lambda: f64) ![]f64 {
    assert(isFinite(lambda));
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.poisson(random, lambda);
    }
    return slice;
}

/// size ∈ {0,1,2,⋯}, prob ∈ [0,1]
pub fn binomial(allocator: Allocator, random: Random, n: usize, size: u64, prob: f64) ![]f64 {
    assert(0 <= prob and prob <= 1);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* =  implementation.binomial(random, size, prob);
    }
    return slice;
}

/// size ∈ {1,2,3,⋯}, prob ∈ (0,1]
pub fn negativeBinomial(allocator: Allocator, random: Random, n: usize, size: u64, prob: f64) ![]f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.negativeBinomial(random, size, prob);
    }
    return slice;
}

/// min and max ∈ (-∞,∞)
pub fn uniform(allocator: Allocator, random: Random, n: usize, min: f64, max: f64) ![]f64 {
    assert(isFinite(min) and isFinite(max));
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.uniform(random, min, max);
    }
    return slice;
}

/// rate ∈ (0,∞)
pub fn exponential(allocator: Allocator, random: Random, n: usize, rate: f64) ![]f64 {
    assert(isFinite(rate));
    assert(rate > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.exponential(random, rate);
    }
    return slice;
}

/// shape and rate ∈ (0,∞)
pub fn weibull(allocator: Allocator, random: Random, n: usize, shape: f64, rate: f64) ![]f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.weibull(random, shape, rate);
    }
    return slice;
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn cauchy(allocator: Allocator, random: Random, n: usize, location: f64, scale: f64) ![]f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.cauchy(random, location, scale);
    }
    return slice;
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn logistic(allocator: Allocator, random: Random, n: usize, location: f64, scale: f64) ![]f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.logistic(random, location, scale);
    }
    return slice;
}

/// shape and rate ∈ (0,∞)
pub fn gamma(allocator: Allocator, random: Random, n: usize, shape: f64, rate: f64) ![]f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.gamma(random, shape, rate);
    }
    return slice;
}

/// df ∈ (0,∞)
pub fn chiSquared(allocator: Allocator, random: Random, n: usize, df: f64) ![]f64 {
    assert(isFinite(df));
    assert(df > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.chiSquared(random, df);
    }
    return slice;
}

/// df1 and df2 ∈ (0,∞)
pub fn f(allocator: Allocator, random: Random, n: usize, df1: f64, df2: f64) ![]f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.f(random, df1, df2);
    }
    return slice;
}

/// shape1 and shape2 ∈ (0,∞)
pub fn beta(allocator: Allocator, random: Random, n: usize, shape1: f64, shape2: f64) ![]f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.beta(random, shape1, shape2);
    }
    return slice;
}

/// mean ∈ (-∞,∞), sd ∈ (0,∞)
pub fn normal(allocator: Allocator, random: Random, n: usize, mean: f64, sd: f64) ![]f64 {
    assert(isFinite(mean) and isFinite(sd));
    assert(sd > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.normal(random, mean, sd);
    }
    return slice;
}

/// meanlog ∈ (-∞,∞), sdlog ∈ (0,∞)
pub fn logNormal(allocator: Allocator, random: Random, n: usize, meanlog: f64, sdlog: f64) ![]f64 {
    assert(isFinite(meanlog) and isFinite(sdlog));
    assert(sdlog > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.logNormal(random, meanlog, sdlog);
    }
    return slice;
}

/// df ∈ (0,∞)
pub fn t(allocator: Allocator, random: Random, n: usize, df: f64) ![]f64 {
    assert(isFinite(df));
    assert(df > 0);
    const slice = try allocator.alloc(f64, n);
    for (slice) |*x| {
        x.* = implementation.t(random, df);
    }
    return slice;
}
