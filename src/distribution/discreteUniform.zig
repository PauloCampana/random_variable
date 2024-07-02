//! Support: {a,⋯,b}
//!
//! Parameters:
//! - a: `min` ∈ {⋯,-1,0,1,⋯}
//! - b: `max` ∈ {a,a + 1,⋯}

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// p(x) = 1 / (b - a + 1)
pub fn density(x: f64, min: i64, max: i64) f64 {
    assert(min <= max);
    assert(!isNan(x));
    const fmin = @as(f64, @floatFromInt(min));
    const fmax = @as(f64, @floatFromInt(max));
    if (x < fmin or x > fmax or x != @round(x)) {
        return 0;
    }
    return 1 / (fmax - fmin + 1);
}

/// F(q) = (⌊q⌋ - a + 1) / (b - a  + 1)
pub fn probability(q: f64, min: i64, max: i64) f64 {
    assert(min <= max);
    assert(!isNan(q));
    const fmin = @as(f64, @floatFromInt(min));
    const fmax = @as(f64, @floatFromInt(max));
    if (q < fmin) {
        return 0;
    }
    if (q >= fmax) {
        return 1;
    }
    return (@floor(q) - fmin + 1) / (fmax - fmin + 1);
}

/// Q(p) = ⌈p (b - a + 1)⌉ + a - 1
pub fn quantile(p: f64, min: i64, max: i64) f64 {
    assert(min <= max);
    assert(0 <= p and p <= 1);
    const fmin = @as(f64, @floatFromInt(min));
    const fmax = @as(f64, @floatFromInt(max));
    if (p == 0) {
        return fmin;
    }
    return @ceil(p * (fmax - fmin + 1)) + fmin - 1;
}

pub fn random(generator: std.Random, min: i64, max: i64) f64 {
    assert(min <= max);
    const uni = generator.intRangeAtMost(i64, min, max);
    return @floatFromInt(uni);
}

pub fn fill(buffer: []f64, generator: std.Random, min: i64, max: i64) []f64 {
    assert(min <= max);
    for (buffer) |*x| {
        const uni = generator.intRangeAtMost(i64, min, max);
        x.* = @floatFromInt(uni);
    }
    return buffer;
}

export fn rv_discrete_uniform_density(x: f64, min: i64, max: i64) f64 {
    return density(x, min, max);
}
export fn rv_discrete_uniform_probability(q: f64, min: i64, max: i64) f64 {
    return probability(q, min, max);
}
export fn rv_discrete_uniform_quantile(p: f64, min: i64, max: i64) f64 {
    return quantile(p, min, max);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 0, 1));
    try expectEqual(0, density( inf, 0, 1));

    try expectApproxEqRel(0    , density( 2, 3, 10), eps);
    try expectApproxEqRel(0.125, density( 3, 3, 10), eps);
    try expectApproxEqRel(0.125, density( 6, 3, 10), eps);
    try expectApproxEqRel(0.125, density(10, 3, 10), eps);
    try expectApproxEqRel(0    , density(11, 3, 10), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0    , probability( 2, 3, 10), eps);
    try expectApproxEqRel(0.125, probability( 3, 3, 10), eps);
    try expectApproxEqRel(0.5  , probability( 6, 3, 10), eps);
    try expectApproxEqRel(1    , probability(10, 3, 10), eps);
    try expectApproxEqRel(1    , probability(11, 3, 10), eps);
}

test quantile {
    try expectEqual( 3, quantile(0    , 3, 10));
    try expectEqual( 3, quantile(0.124, 3, 10));
    try expectEqual( 3, quantile(0.125, 3, 10));
    try expectEqual( 4, quantile(0.126, 3, 10));
    try expectEqual( 4, quantile(0.249, 3, 10));
    try expectEqual( 4, quantile(0.250, 3, 10));
    try expectEqual( 5, quantile(0.251, 3, 10));
    try expectEqual(10, quantile(1    , 3, 10));
}
