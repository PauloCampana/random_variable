//! Support: X ∈ [a,b]
//!
//! Parameters:
//! - a: `min` ∈ (-∞,∞)
//! - b: `max` ∈ [ a,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = false;
pub const parameters = 2;

/// f(x) = 1 / (b - a).
pub fn density(x: f64, min: f64, max: f64) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(min <= max);
    assert(!isNan(x));
    if (x < min or x > max) {
        return 0;
    }
    return 1 / (max - min);
}

/// F(q) = (q - a) / (b - a).
pub fn probability(q: f64, min: f64, max: f64) f64 {
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

/// Q(p) = a + (b - a)p.
pub fn quantile(p: f64, min: f64, max: f64) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(min <= max);
    assert(0 <= p and p <= 1);
    return min + (max - min) * p;
}

/// Uses the quantile function.
pub const random = struct {
    pub fn single(generator: std.rand.Random, min: f64, max: f64) f64 {
        assert(isFinite(min) and isFinite(max));
        assert(min <= max);
        const uni = generator.float(f64);
        return min + (max - min) * uni;
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, min: f64, max: f64) []f64 {
        assert(isFinite(min) and isFinite(max));
        assert(min <= max);
        const scale = max - min;
        for (buffer) |*x| {
            const uni = generator.float(f64);
            x.* = min + scale * uni;
        }
        return buffer;
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "uniform.density" {
    try expectEqual(0, density(-inf, 0, 1));
    try expectEqual(0, density( inf, 0, 1));

    try expectApproxEqRel(0  , density(2, 3, 5), eps);
    try expectApproxEqRel(0.5, density(3, 3, 5), eps);
    try expectApproxEqRel(0.5, density(4, 3, 5), eps);
    try expectApproxEqRel(0.5, density(5, 3, 5), eps);
    try expectApproxEqRel(0  , density(6, 3, 5), eps);
}

test "uniform.probability" {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0  , probability(3  , 3, 5), eps);
    try expectApproxEqRel(0.2, probability(3.4, 3, 5), eps);
    try expectApproxEqRel(0.4, probability(3.8, 3, 5), eps);
    try expectApproxEqRel(0.6, probability(4.2, 3, 5), eps);
    try expectApproxEqRel(0.8, probability(4.6, 3, 5), eps);
    try expectApproxEqRel(1  , probability(5  , 3, 5), eps);
}

test "uniform.quantile" {
    try expectApproxEqRel(3  , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(3.4, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(3.8, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(4.2, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(4.6, quantile(0.8, 3, 5), eps);
    try expectApproxEqRel(5  , quantile(1  , 3, 5), eps);
}

test "uniform.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.75d61490b23dfp-2, random.single(gen, 0, 1), eps);
    try expectApproxEqRel(0x1.a6f3dc380d507p-2, random.single(gen, 0, 1), eps);
    try expectApproxEqRel(0x1.fdf91ec9a7bfcp-2, random.single(gen, 0, 1), eps);
    try expectApproxEqRel(0x1.0000000000000p+0, random.single(gen, 1, 1), eps);
}
