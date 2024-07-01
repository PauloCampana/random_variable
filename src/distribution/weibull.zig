//! Support: X ∈ [0,∞)
//!
//! Parameters:
//! - α: `shape` ∈ (0,∞)
//! - λ: `rate`  ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = αλ (λx)^(α - 1) exp(-(λx)^α).
pub fn density(x: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(x));
    if (x < 0 or x == inf) {
        return 0;
    }
    if (x == 0) {
        if (shape == 1) {
            return rate;
        }
        return if (shape < 1) inf else 0;
    }
    const z = rate * x;
    const zam1 = std.math.pow(f64, z, shape - 1);
    const za = zam1 * z;
    return shape * rate * zam1 * @exp(-za);
}

/// F(q) = 1 - exp(-(λq)^α).
pub fn probability(q: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = rate * q;
    const za = std.math.pow(f64, z, shape);
    return -std.math.expm1(-za);
}

/// Q(p) = (-ln(1 - p))^(1 / α) / λ.
pub fn quantile(p: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(0 <= p and p <= 1);
    const q1 = -std.math.log1p(-p);
    const q2 = std.math.pow(f64, q1, 1 / shape);
    return q2 / rate;
}

pub fn random(generator: std.Random, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    const exp = generator.floatExp(f64);
    const wei = std.math.pow(f64, exp, 1 / shape);
    return wei / rate;
}

pub fn fill(buffer: []f64, generator: std.Random, shape: f64, rate: f64) []f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    const invshape = 1 / shape;
    for (buffer) |*x| {
        const exp = generator.floatExp(f64);
        const wei = std.math.pow(f64, exp, invshape);
        x.* = wei / rate;
    }
    return buffer;
}

export fn rv_weibull_density(x: f64, shape: f64, rate: f64) f64 {
    return density(x, shape, rate);
}
export fn rv_weibull_probability(q: f64, shape: f64, rate: f64) f64 {
    return probability(q, shape, rate);
}
export fn rv_weibull_quantile(p: f64, shape: f64, rate: f64) f64 {
    return quantile(p, shape, rate);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3, 0.5));
    try expectEqual(0, density( inf, 3, 0.5));

    try expectEqual(inf, density(0, 0.9, 5));
    try expectEqual(5  , density(0, 1  , 5));
    try expectEqual(0  , density(0, 1.1, 5));

    try expectApproxEqRel(0                 , density(0, 3, 0.5), eps);
    try expectApproxEqRel(0.3309363384692233, density(1, 3, 0.5), eps);
    try expectApproxEqRel(0.5518191617571635, density(2, 3, 0.5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3, 0.5));
    try expectEqual(1, probability( inf, 3, 0.5));

    try expectApproxEqRel(0                 , probability(0, 3, 0.5), eps);
    try expectApproxEqRel(0.1175030974154046, probability(1, 3, 0.5), eps);
    try expectApproxEqRel(0.6321205588285577, probability(2, 3, 0.5), eps);
}

test quantile {
    try expectApproxEqRel(0                , quantile(0  , 3, 0.5), eps);
    try expectApproxEqRel(1.213085586248216, quantile(0.2, 3, 0.5), eps);
    try expectApproxEqRel(1.598775754926823, quantile(0.4, 3, 0.5), eps);
    try expectApproxEqRel(1.942559933595852, quantile(0.6, 3, 0.5), eps);
    try expectApproxEqRel(2.343804613759100, quantile(0.8, 3, 0.5), eps);
    try expectEqual      (inf              , quantile(1  , 3, 0.5)     );
}
