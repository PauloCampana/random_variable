//! Support: [0,∞)
//!
//! Parameters:
//! - σ: `scale` ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = exp(-x / σ) / σ
pub fn density(x: f64, scale: f64) f64 {
    assert(isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    if (x < 0) {
        return 0;
    }
    const z = x / scale;
    return @exp(-z) / scale;
}

/// F(q) = 1 - exp(-q / σ)
pub fn probability(q: f64, scale: f64) f64 {
    assert(isFinite(scale));
    assert(scale > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = q / scale;
    return -std.math.expm1(-z);
}

/// S(t) = exp(-t / σ)
pub fn survival(t: f64, scale: f64) f64 {
    assert(isFinite(scale));
    assert(scale > 0);
    assert(!isNan(t));
    if (t <= 0) {
        return 1;
    }
    const z = t / scale;
    return @exp(-z);
}

/// Q(p) = -σ ln(1 - p)
pub fn quantile(p: f64, scale: f64) f64 {
    assert(isFinite(scale));
    assert(scale > 0);
    assert(0 <= p and p <= 1);
    const q = -std.math.log1p(-p);
    return scale * q;
}

pub fn random(generator: std.Random, scale: f64) f64 {
    assert(isFinite(scale));
    assert(scale > 0);
    const exp = generator.floatExp(f64);
    return scale * exp;
}

pub fn fill(buffer: []f64, generator: std.Random, scale: f64) void {
    assert(isFinite(scale));
    assert(scale > 0);
    for (buffer) |*x| {
        const exp = generator.floatExp(f64);
        x.* = scale * exp;
    }
}

export fn rv_exponential_density(x: f64, scale: f64) f64 {
    return density(x, scale);
}
export fn rv_exponential_probability(q: f64, scale: f64) f64 {
    return probability(q, scale);
}
export fn rv_exponential_survival(t: f64, scale: f64) f64 {
    return survival(t, scale);
}
export fn rv_exponential_quantile(p: f64, scale: f64) f64 {
    return quantile(p, scale);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3));
    try expectEqual(0, density( inf, 3));

    try expectApproxEqRel(0.3333333333333333, density(0, 3), eps);
    try expectApproxEqRel(0.2388437701912630, density(1, 3), eps);
    try expectApproxEqRel(0.1711390396775306, density(2, 3), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3));
    try expectEqual(1, probability( inf, 3));

    try expectApproxEqRel(0                 , probability(0, 3), eps);
    try expectApproxEqRel(0.2834686894262107, probability(1, 3), eps);
    try expectApproxEqRel(0.4865828809674079, probability(2, 3), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 3));
    try expectEqual(0, survival( inf, 3));

    try expectApproxEqRel(1                 , survival(0, 3), eps);
    try expectApproxEqRel(0.7165313105737892, survival(1, 3), eps);
    try expectApproxEqRel(0.5134171190325920, survival(2, 3), eps);
}

test quantile {
    try expectApproxEqRel(0                 , quantile(0  , 3), eps);
    try expectApproxEqRel(0.6694306539426292, quantile(0.2, 3), eps);
    try expectApproxEqRel(1.5324768712979720, quantile(0.4, 3), eps);
    try expectApproxEqRel(2.7488721956224651, quantile(0.6, 3), eps);
    try expectApproxEqRel(4.8283137373023011, quantile(0.8, 3), eps);
    try expectEqual      (inf               , quantile(1  , 3)     );
}
