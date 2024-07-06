//! Support: [0,∞)
//!
//! Parameters:
//! - α: `shape` ∈ (0,∞)
//! - σ: `scale` ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = α / σ (x / σ)^(α - 1) exp(-(x / σ)^α)
pub fn density(x: f64, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    assert(!isNan(x));
    if (x < 0 or x == inf) {
        return 0;
    }
    if (x == 0) {
        if (shape == 1) {
            return 1 / scale;
        }
        return if (shape < 1) inf else 0;
    }
    const z = x / scale;
    const zam1 = std.math.pow(f64, z, shape - 1);
    const za = zam1 * z;
    return shape / scale * zam1 * @exp(-za);
}

/// F(q) = 1 - exp(-(q / σ)^α)
pub fn probability(q: f64, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = q / scale;
    const za = std.math.pow(f64, z, shape);
    return -std.math.expm1(-za);
}

/// Q(p) = σ (-ln(1 - p))^(1 / α)
pub fn quantile(p: f64, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    assert(0 <= p and p <= 1);
    const exp = -std.math.log1p(-p);
    const wei = std.math.pow(f64, exp, 1 / shape);
    return scale * wei;
}

pub fn random(generator: std.Random, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    const exp = generator.floatExp(f64);
    const wei = std.math.pow(f64, exp, 1 / shape);
    return scale * wei;
}

pub fn fill(buffer: []f64, generator: std.Random, shape: f64, scale: f64) void {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    const invshape = 1 / shape;
    for (buffer) |*x| {
        const exp = generator.floatExp(f64);
        const wei = std.math.pow(f64, exp, invshape);
        x.* = scale * wei;
    }
}

export fn rv_weibull_density(x: f64, shape: f64, scale: f64) f64 {
    return density(x, shape, scale);
}
export fn rv_weibull_probability(q: f64, shape: f64, scale: f64) f64 {
    return probability(q, shape, scale);
}
export fn rv_weibull_quantile(p: f64, shape: f64, scale: f64) f64 {
    return quantile(p, shape, scale);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectEqual(inf, density(0, 0.9, 5));
    try expectEqual(0.2, density(0, 1  , 5));
    try expectEqual(0  , density(0, 1.1, 5));

    try expectApproxEqRel(0                  , density(0, 3, 5), eps);
    try expectApproxEqRel(0.02380876595608945, density(1, 3, 5), eps);
    try expectApproxEqRel(0.09004847995495003, density(2, 3, 5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                   , probability(0, 3, 5), eps);
    try expectApproxEqRel(0.007968085162939369, probability(1, 3, 5), eps);
    try expectApproxEqRel(0.061995000469270512, probability(2, 3, 5), eps);
}

test quantile {
    try expectApproxEqRel(0                , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(3.032713965620540, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(3.996939387317056, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(4.856399833989629, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(5.859511534397749, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf              , quantile(1  , 3, 5)     );
}
