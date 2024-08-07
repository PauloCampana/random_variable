//! Support: [0,∞)
//!
//! Parameters:
//! - μ: `log_location` ∈ (-∞,∞)
//! - σ: `log_scale`    ∈ ( 0,∞)

const std = @import("std");
const normal = @import("normal.zig");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = exp(-((ln(x) - μ) / σ)^2 / 2) / (xσ sqrt(2π))
pub fn density(x: f64, log_location: f64, log_scale: f64) f64 {
    assert(isFinite(log_location) and isFinite(log_scale));
    assert(log_scale > 0);
    assert(!isNan(x));
    if (x <= 0) {
        return 0;
    }
    const z = (@log(x) - log_location) / log_scale;
    const sqrt2pi = comptime @sqrt(2 * std.math.pi);
    return @exp(-0.5 * z * z) / (x * log_scale * sqrt2pi);
}

/// No closed form
pub fn probability(q: f64, log_location: f64, log_scale: f64) f64 {
    if (q <= 0) {
        return 0;
    }
    return normal.probability(@log(q), log_location, log_scale);
}

/// No closed form
pub fn survival(t: f64, log_location: f64, log_scale: f64) f64 {
    if (t <= 0) {
        return 1;
    }
    return normal.survival(@log(t), log_location, log_scale);
}

/// No closed form
pub fn quantile(p: f64, log_location: f64, log_scale: f64) f64 {
    const q = normal.quantile(p, log_location, log_scale);
    return @exp(q);
}

pub fn random(generator: std.Random, log_location: f64, log_scale: f64) f64 {
    assert(isFinite(log_location) and isFinite(log_scale));
    assert(log_scale > 0);
    const nor = generator.floatNorm(f64);
    return @exp(log_location + log_scale * nor);
}

pub fn fill(buffer: []f64, generator: std.Random, log_location: f64, log_scale: f64) void {
    assert(isFinite(log_location) and isFinite(log_scale));
    assert(log_scale > 0);
    for (buffer) |*x| {
        const nor = generator.floatNorm(f64);
        x.* = @exp(log_location + log_scale * nor);
    }
}

export fn rv_log_normal_density(x: f64, log_location: f64, log_scale: f64) f64 {
    return density(x, log_location, log_scale);
}
export fn rv_log_normal_probability(q: f64, log_location: f64, log_scale: f64) f64 {
    return probability(q, log_location, log_scale);
}
export fn rv_log_normal_survival(t: f64, log_location: f64, log_scale: f64) f64 {
    return survival(t, log_location, log_scale);
}
export fn rv_log_normal_quantile(p: f64, log_location: f64, log_scale: f64) f64 {
    return quantile(p, log_location, log_scale);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 0, 1));
    try expectEqual(0, density( inf, 0, 1));

    try expectApproxEqRel(0                 , density(0, 0, 1), eps);
    try expectApproxEqRel(0.3989422804014327, density(1, 0, 1), eps);
    try expectApproxEqRel(0.1568740192789811, density(2, 0, 1), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0                 , probability(0, 0, 1), eps);
    try expectApproxEqRel(0.5               , probability(1, 0, 1), eps);
    try expectApproxEqRel(0.7558914042144173, probability(2, 0, 1), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 0, 1));
    try expectEqual(0, survival( inf, 0, 1));

    try expectApproxEqRel(1                 , survival(0, 0, 1), eps);
    try expectApproxEqRel(0.5               , survival(1, 0, 1), eps);
    try expectApproxEqRel(0.2441085957855827, survival(2, 0, 1), eps);
}

test quantile {
    try expectApproxEqRel(0                 , quantile(0  , 0, 1), eps);
    try expectApproxEqRel(0.4310111868818386, quantile(0.2, 0, 1), eps);
    try expectApproxEqRel(0.7761984141563506, quantile(0.4, 0, 1), eps);
    try expectApproxEqRel(1.2883303827500079, quantile(0.6, 0, 1), eps);
    try expectApproxEqRel(2.3201253945043181, quantile(0.8, 0, 1), eps);
    try expectEqual      (inf               , quantile(1  , 0, 1)     );
}
