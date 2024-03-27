//! Support: X ∈ [0,1]
//!
//! Parameters:
//! - α: `shape1` ∈ (0,∞)
//! - β: `shape2` ∈ (0,∞)

const std = @import("std");
const gamma = @import("gamma.zig");
const special = @import("../special.zig");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = false;

/// f(x) = x^(α - 1) (1 - x)^(β - 1) / beta(α, β).
pub fn density(x: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(x));
    if (x < 0 or x > 1) {
        return 0;
    }
    if (x == 0) {
        if (shape1 == 1) {
            return shape2;
        }
        return if (shape1 < 1) inf else 0;
    }
    if (x == 1) {
        if (shape2 == 1) {
            return shape1;
        }
        return if (shape2 < 1) inf else 0;
    }
    const num = (shape1 - 1) * @log(x) + (shape2 - 1) * std.math.log1p(-x);
    const den = special.lbeta(shape1, shape2);
    return @exp(num - den);
}

/// No closed form.
pub fn probability(q: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    if (q >= 1) {
        return 1;
    }
    return special.beta_probability(shape1, shape2, q);
}

/// No closed form.
pub fn quantile(p: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(0 <= p and p <= 1);
    return special.beta_quantile(shape1, shape2, p);
}

pub fn random(generator: std.Random, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    if (shape1 == 1) {
        const uni = generator.float(f64);
        return 1 - std.math.pow(f64, uni, 1 / shape2);
    }
    if (shape2 == 1) {
        const uni = generator.float(f64);
        return std.math.pow(f64, uni, 1 / shape1);
    }
    if (shape1 < 1 and shape2 < 1) {
        return rejection(generator, 1 / shape1, 1 / shape2);
    }
    const gam1 = gamma.random(generator, shape1, 1);
    const gam2 = gamma.random(generator, shape2, 1);
    return gam1 / (gam1 + gam2);
}

pub fn fill(buffer: []f64, generator: std.Random, shape1: f64, shape2: f64) []f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    const inva = 1 / shape1;
    const invb = 1 / shape2;
    if (shape1 == 1) {
        for (buffer) |*x| {
            const uni = generator.float(f64);
            x.* = 1 - std.math.pow(f64, uni, invb);
        }
        return buffer;
    }
    if (shape2 == 1) {
        for (buffer) |*x| {
            const uni = generator.float(f64);
            x.* = std.math.pow(f64, uni, inva);
        }
        return buffer;
    }
    if (shape1 < 1 and shape2 < 1) {
        for (buffer) |*x| {
            x.* = rejection(generator, inva, invb);
        }
        return buffer;
    }
    for (buffer) |*x| {
        const gam1 = gamma.random(generator, shape1, 1);
        const gam2 = gamma.random(generator, shape2, 1);
        x.* = gam1 / (gam1 + gam2);
    }
    return buffer;
}

// http://luc.devroye.org/chapter_nine.pdf page 416.
fn rejection(generator: std.Random, inva: f64, invb: f64) f64 {
    while (true) {
        const uni1 = generator.float(f64);
        const uni2 = generator.float(f64);
        const x = std.math.pow(f64, uni1, inva);
        const y = std.math.pow(f64, uni2, invb);
        const z = x + y;
        if (z <= 1) {
            return x / z;
        }
    }
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "beta.density" {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectEqual(inf, density(0, 0.9, 5  ));
    try expectEqual(  5, density(0, 1  , 5  ));
    try expectEqual(  0, density(0, 1.1, 5  ));
    try expectEqual(inf, density(1, 3  , 0.9));
    try expectEqual(  3, density(1, 3  , 1  ));
    try expectEqual(  0, density(1, 3  , 1.1));

    try expectApproxEqRel(0      , density(0  , 3, 5), eps);
    try expectApproxEqRel(1.72032, density(0.2, 3, 5), eps);
    try expectApproxEqRel(0.10752, density(0.8, 3, 5), eps);
    try expectApproxEqRel(0      , density(1  , 3, 5), eps);
}

test "beta.probability" {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0       , probability(0  , 3, 5), eps);
    try expectApproxEqRel(0.148032, probability(0.2, 3, 5), eps);
    try expectApproxEqRel(0.995328, probability(0.8, 3, 5), eps);
    try expectApproxEqRel(1       , probability(1  , 3, 5), eps);
}

test "beta.quantile" {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.2283264643498391, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.3205858305642004, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(0.4092151219095550, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(0.5167577700975785, quantile(0.8, 3, 5), eps);
    try expectApproxEqRel(1                 , quantile(1  , 3, 5), eps);
}
