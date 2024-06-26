//! Support: [0,∞)
//!
//! Parameters:
//! - λ: `rate` ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = λ exp(-λx)
pub fn density(x: f64, rate: f64) f64 {
    assert(isFinite(rate));
    assert(rate > 0);
    assert(!isNan(x));
    if (x < 0) {
        return 0;
    }
    return rate * @exp(-rate * x);
}

/// F(q) = 1 - exp(-λq)
pub fn probability(q: f64, rate: f64) f64 {
    assert(isFinite(rate));
    assert(rate > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = rate * q;
    return -std.math.expm1(-z);
}

/// Q(p) = -ln(1 - p) / λ
pub fn quantile(p: f64, rate: f64) f64 {
    assert(isFinite(rate));
    assert(rate > 0);
    assert(0 <= p and p <= 1);
    const q = -std.math.log1p(-p);
    return q / rate;
}

pub fn random(generator: std.Random, rate: f64) f64 {
    assert(isFinite(rate));
    assert(rate > 0);
    const exp = generator.floatExp(f64);
    return exp / rate;
}

pub fn fill(buffer: []f64, generator: std.Random, rate: f64) void {
    assert(isFinite(rate));
    assert(rate > 0);
    for (buffer) |*x| {
        const exp = generator.floatExp(f64);
        x.* = exp / rate;
    }
}

export fn rv_exponential_density(x: f64, rate: f64) f64 {
    return density(x, rate);
}
export fn rv_exponential_probability(q: f64, rate: f64) f64 {
    return probability(q, rate);
}
export fn rv_exponential_quantile(p: f64, rate: f64) f64 {
    return quantile(p, rate);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3));
    try expectEqual(0, density( inf, 3));

    try expectApproxEqRel(3                   , density(0, 3), eps);
    try expectApproxEqRel(0.149361205103591900, density(1, 3), eps);
    try expectApproxEqRel(0.007436256529999075, density(2, 3), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3));
    try expectEqual(1, probability( inf, 3));

    try expectApproxEqRel(0                 , probability(0, 3), eps);
    try expectApproxEqRel(0.9502129316321360, probability(1, 3), eps);
    try expectApproxEqRel(0.9975212478233336, probability(2, 3), eps);
}

test quantile {
    try expectApproxEqRel(0                  , quantile(0  , 3), eps);
    try expectApproxEqRel(0.07438118377140325, quantile(0.2, 3), eps);
    try expectApproxEqRel(0.17027520792199691, quantile(0.4, 3), eps);
    try expectApproxEqRel(0.30543024395805174, quantile(0.6, 3), eps);
    try expectApproxEqRel(0.53647930414470013, quantile(0.8, 3), eps);
    try expectEqual      (inf                , quantile(1  , 3)     );
}
