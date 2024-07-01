//! Support: X ∈ [0,∞)
//!
//! Parameters:
//! - p: `shape1` ∈ (0,∞)
//! - α: `shape2` ∈ (0,∞)
//! - σ: `scale`  ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = pα/σ (x / σ)^(pα - 1) / (1 + (x / σ)^α)^(p + 1).
pub fn density(x: f64, shape1: f64, shape2: f64, scale: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2) and isFinite(scale));
    assert(shape1 > 0 and shape2 > 0 and scale > 0);
    assert(!isNan(x));
    if (x < 0 or x == inf) {
        return 0;
    }
    if (x == 0) {
        const prod = shape1 * shape2;
        if (prod == 1) {
            return 1 / scale;
        }
        return if (prod < 1) inf else 0;
    }
    const z = x / scale;
    const inner = 1 + std.math.pow(f64, z, shape2);
    const num = std.math.pow(f64, z, shape1 * shape2 - 1);
    const den = std.math.pow(f64, inner, shape1 + 1);
    return shape1 * shape2 / scale * num / den;
}

/// F(q) = (1 + (q / σ)^-α)^-p.
pub fn probability(q: f64, shape1: f64, shape2: f64, scale: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2) and isFinite(scale));
    assert(shape1 > 0 and shape2 > 0 and scale > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = q / scale;
    const inner = 1 + std.math.pow(f64, z, -shape2);
    return std.math.pow(f64, inner, -shape1);
}

/// Q(x) = σ(x^(-1 / p) - 1)^(- 1 / α)
pub fn quantile(p: f64, shape1: f64, shape2: f64, scale: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2) and isFinite(scale));
    assert(shape1 > 0 and shape2 > 0 and scale > 0);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return 0;
    }
    if (p == 1) {
        return inf;
    }
    const base = std.math.pow(f64, p, -1 / shape1) - 1;
    const pow = std.math.pow(f64, base, -1 / shape2);
    return scale * pow;
}

pub fn random(generator: std.Random, shape1: f64, shape2: f64, scale: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2) and isFinite(scale));
    assert(shape1 > 0 and shape2 > 0 and scale > 0);
    const uni = generator.float(f64);
    const base = std.math.pow(f64, uni, -1 / shape1) - 1;
    const pow = std.math.pow(f64, base, -1 / shape2);
    return scale * pow;
}

pub fn fill(buffer: []f64, generator: std.Random, shape1: f64, shape2: f64, scale: f64) []f64 {
    assert(isFinite(shape1) and isFinite(shape2) and isFinite(scale));
    assert(shape1 > 0 and shape2 > 0 and scale > 0);
    const minvshape1 = -1 / shape1;
    const minvshape2 = -1 / shape2;
    for (buffer) |*x| {
        const uni = generator.float(f64);
        const base = std.math.pow(f64, uni, minvshape1) - 1;
        const pow = std.math.pow(f64, base, minvshape2);
        x.* = scale * pow;
    }
    return buffer;
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3, 5, 1));
    try expectEqual(0, density( inf, 3, 5, 1));

    try expectEqual(inf, density(0, 0.9, 0.9, 1));
    try expectEqual(inf, density(0, 0.9, 1  , 1));
    try expectEqual(inf, density(0, 0.9, 1.1, 1));
    try expectEqual(inf, density(0, 1  , 0.9, 1));
    try expectEqual(  1, density(0, 1  , 1  , 1));
    try expectEqual(  0, density(0, 1  , 1.1, 1));
    try expectEqual(inf, density(0, 1.1, 0.9, 1));
    try expectEqual(  0, density(0, 1.1, 1  , 1));
    try expectEqual(  0, density(0, 1.1, 1.1, 1));

    try expectApproxEqRel(0                 , density(0, 3, 5, 1), eps);
    try expectApproxEqRel(0.9375            , density(1, 3, 5, 1), eps);
    try expectApproxEqRel(0.2072313417166910, density(2, 3, 5, 1), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3, 5, 1));
    try expectEqual(1, probability( inf, 3, 5, 1));

    try expectApproxEqRel(0                 , probability(0, 3, 5, 1), eps);
    try expectApproxEqRel(0.125             , probability(1, 3, 5, 1), eps);
    try expectApproxEqRel(0.9118179035534407, probability(2, 3, 5, 1), eps);
}

test quantile {
    try expectApproxEqRel(0                , quantile(0  , 3, 5, 1), eps);
    try expectApproxEqRel(1.070905805432601, quantile(0.2, 3, 5, 1), eps);
    try expectApproxEqRel(1.228614306456529, quantile(0.4, 3, 5, 1), eps);
    try expectApproxEqRel(1.400457235365702, quantile(0.6, 3, 5, 1), eps);
    try expectApproxEqRel(1.669002652849758, quantile(0.8, 3, 5, 1), eps);
    try expectEqual      (inf              , quantile(1  , 3, 5, 1)     );
}
