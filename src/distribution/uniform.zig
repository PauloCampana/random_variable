//! Support: [a,b]
//!
//! Parameters:
//! - a: `min` ∈ (-∞,∞)
//! - b: `max` ∈ [ a,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = 1 / (b - a)
pub fn density(x: f64, min: f64, max: f64) callconv(.c) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(min <= max);
    assert(!isNan(x));
    if (x < min or x > max) {
        return 0;
    }
    return 1 / (max - min);
}

/// F(q) = (q - a) / (b - a)
pub fn probability(q: f64, min: f64, max: f64) callconv(.c) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(min <= max);
    assert(!isNan(q));
    if (q <= min) {
        return 0;
    }
    if (q >= max) {
        return 1;
    }
    return (q - min) / (max - min);
}

/// S(t) = (b - t) / (b - a)
pub fn survival(t: f64, min: f64, max: f64) callconv(.c) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(min <= max);
    assert(!isNan(t));
    if (t <= min) {
        return 1;
    }
    if (t >= max) {
        return 0;
    }
    return (max - t) / (max - min);
}

/// Q(p) = a + (b - a)p
pub fn quantile(p: f64, min: f64, max: f64) callconv(.c) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(min <= max);
    assert(0 <= p and p <= 1);
    return min + (max - min) * p;
}

pub fn random(generator: std.Random, min: f64, max: f64) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(min <= max);
    const uni = generator.float(f64);
    return min + (max - min) * uni;
}

pub fn fill(buffer: []f64, generator: std.Random, min: f64, max: f64) void {
    assert(isFinite(min) and isFinite(max));
    assert(min <= max);
    const scale = max - min;
    for (buffer) |*x| {
        const uni = generator.float(f64);
        x.* = min + scale * uni;
    }
}

comptime {
    @export(&density, .{ .name = "rv_uniform_density" });
    @export(&probability, .{ .name = "rv_uniform_probability" });
    @export(&survival, .{ .name = "rv_uniform_survival" });
    @export(&quantile, .{ .name = "rv_uniform_quantile" });
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 0, 1));
    try expectEqual(0, density( inf, 0, 1));

    try expectApproxEqRel(0  , density(2, 3, 5), eps);
    try expectApproxEqRel(0.5, density(3, 3, 5), eps);
    try expectApproxEqRel(0.5, density(4, 3, 5), eps);
    try expectApproxEqRel(0.5, density(5, 3, 5), eps);
    try expectApproxEqRel(0  , density(6, 3, 5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0  , probability(3  , 3, 5), eps);
    try expectApproxEqRel(0.2, probability(3.4, 3, 5), eps);
    try expectApproxEqRel(0.4, probability(3.8, 3, 5), eps);
    try expectApproxEqRel(0.6, probability(4.2, 3, 5), eps);
    try expectApproxEqRel(0.8, probability(4.6, 3, 5), eps);
    try expectApproxEqRel(1  , probability(5  , 3, 5), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 0, 1));
    try expectEqual(0, survival( inf, 0, 1));

    try expectApproxEqRel(1  , survival(3  , 3, 5), eps);
    try expectApproxEqRel(0.8, survival(3.4, 3, 5), eps);
    try expectApproxEqRel(0.6, survival(3.8, 3, 5), eps);
    try expectApproxEqRel(0.4, survival(4.2, 3, 5), eps);
    try expectApproxEqRel(0.2, survival(4.6, 3, 5), eps);
    try expectApproxEqRel(0  , survival(5  , 3, 5), eps);
}

test quantile {
    try expectApproxEqRel(3  , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(3.4, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(3.8, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(4.2, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(4.6, quantile(0.8, 3, 5), eps);
    try expectApproxEqRel(5  , quantile(1  , 3, 5), eps);
}
