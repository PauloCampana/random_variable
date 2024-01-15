//! Support: X ∈ {0,1,2,∞}
//!
//! Parameters:
//! - p: `prob` ∈ (0,1]

const std = @import("std");
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = true;
pub const parameters = 1;

/// p(x) = p (1 - p)^x.
pub fn density(x: f64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(!isNan(x));
    if (x < 0 or x != @round(x)) {
        return 0;
    }
    return prob * std.math.pow(f64, (1 - prob), x);
}

/// F(q) = 1 - (1 - p)^(⌊q⌋ + 1).
pub fn probability(q: f64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    const p = (@floor(q) + 1) * std.math.log1p(-prob);
    return -std.math.expm1(p);
}

/// Q(x) = ⌊ln(1 - x) / ln(1 - p)⌋.
pub fn quantile(p: f64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(0 <= p and p <= 1);
    if (p == 1) {
        return inf;
    }
    if (p <= prob) {
        return 0;
    }
    return @floor(std.math.log1p(-p) / std.math.log1p(-prob));
}

/// Uses the relation to Exponential distribution.
pub const random = struct {
    pub fn single(generator: std.rand.Random, prob: f64) f64 {
        assert(0 < prob and prob <= 1);
        const rate = -std.math.log1p(-prob);
        const exp = generator.floatExp(f64);
        return @trunc(exp / rate);
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, prob: f64) []f64 {
        assert(0 < prob and prob <= 1);
        const rate = -std.math.log1p(-prob);
        for (buffer) |*x| {
            const exp = generator.floatExp(f64);
            x.* = @trunc(exp / rate);
        }
        return buffer;
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "geometric.density" {
    try expectEqual(0, density(-inf, 0.2));
    try expectEqual(0, density( inf, 0.2));
    try expectEqual(0, density(-inf, 1  ));
    try expectEqual(0, density( inf, 1  ));

    try expectEqual(1, density(0, 1));
    try expectEqual(0, density(1, 1));

    try expectApproxEqRel(0   , density(-0.1, 0.2), eps);
    try expectApproxEqRel(0.2 , density( 0  , 0.2), eps);
    try expectApproxEqRel(0   , density( 0.1, 0.2), eps);
    try expectApproxEqRel(0   , density( 0.9, 0.2), eps);
    try expectApproxEqRel(0.16, density( 1  , 0.2), eps);
    try expectApproxEqRel(0   , density( 1.1, 0.2), eps);
}

test "geometric.probability" {
    try expectEqual(0, probability(-inf, 0.2));
    try expectEqual(1, probability( inf, 0.2));
    try expectEqual(0, probability(-inf, 1  ));
    try expectEqual(1, probability( inf, 1  ));

    try expectApproxEqRel(0   , probability(-0.1, 0.2), eps);
    try expectApproxEqRel(0.2 , probability( 0  , 0.2), eps);
    try expectApproxEqRel(0.2 , probability( 0.1, 0.2), eps);
    try expectApproxEqRel(0.2 , probability( 0.9, 0.2), eps);
    try expectApproxEqRel(0.36, probability( 1  , 0.2), eps);
    try expectApproxEqRel(0.36, probability( 1.1, 0.2), eps);
}

test "geometric.quantile" {
    try expectEqual(  0, quantile(0   , 0.2));
    try expectEqual(  0, quantile(0.19, 0.2));
    try expectEqual(  0, quantile(0.2 , 0.2));
    try expectEqual(  1, quantile(0.21, 0.2));
    try expectEqual(  1, quantile(0.35, 0.2));
    try expectEqual(  1, quantile(0.36, 0.2));
    try expectEqual(  2, quantile(0.37, 0.2));
    try expectEqual(inf, quantile(1   , 0.2));
}

test "geometric.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(0, random.single(gen, 1));
    try expectEqual(0, random.single(gen, 1));
    try expectEqual(0, random.single(gen, 1));

    try expectEqual(0, random.single(gen, 0.2));
    try expectEqual(1, random.single(gen, 0.2));
    try expectEqual(0, random.single(gen, 0.2));
}
