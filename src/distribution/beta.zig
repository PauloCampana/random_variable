//! Support: [0,1]
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

/// f(x) = x^(α - 1) (1 - x)^(β - 1) / beta(α, β)
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

/// No closed form
pub fn probability(q: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(q));
    return special.beta.probability(q, shape1, shape2);
}

/// No closed form
pub fn survival(t: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(t));
    return special.beta.probability(1 - t, shape2, shape1);
}

/// No closed form
pub fn quantile(p: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(0 <= p and p <= 1);
    return special.beta.quantile(p, shape1, shape2);
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

pub fn fill(buffer: []f64, generator: std.Random, shape1: f64, shape2: f64) void {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    const invshape2 = 1 / shape2;
    const invshape1 = 1 / shape1;
    if (shape1 == 1) {
        for (buffer) |*x| {
            const uni = generator.float(f64);
            x.* = 1 - std.math.pow(f64, uni, invshape2);
        }
        return;
    }
    if (shape2 == 1) {
        for (buffer) |*x| {
            const uni = generator.float(f64);
            x.* = std.math.pow(f64, uni, invshape1);
        }
        return;
    }
    if (shape1 < 1 and shape2 < 1) {
        for (buffer) |*x| {
            x.* = rejection(generator, invshape1, invshape2);
        }
        return;
    }
    for (buffer) |*x| {
        const gam1 = gamma.random(generator, shape1, 1);
        const gam2 = gamma.random(generator, shape2, 1);
        x.* = gam1 / (gam1 + gam2);
    }
}

// http://luc.devroye.org/chapter_nine.pdf page 416.
fn rejection(generator: std.Random, invshape1: f64, invshape2: f64) f64 {
    while (true) {
        const uni1 = generator.float(f64);
        const uni2 = generator.float(f64);
        const x = std.math.pow(f64, uni1, invshape1);
        const y = std.math.pow(f64, uni2, invshape2);
        const z = x + y;
        if (z <= 1) {
            return x / z;
        }
    }
}

export fn rv_beta_density(x: f64, shape1: f64, shape2: f64) f64 {
    return density(x, shape1, shape2);
}
export fn rv_beta_probability(q: f64, shape1: f64, shape2: f64) f64 {
    return probability(q, shape1, shape2);
}
export fn rv_beta_survival(t: f64, shape1: f64, shape2: f64) f64 {
    return survival(t, shape1, shape2);
}
export fn rv_beta_quantile(p: f64, shape1: f64, shape2: f64) f64 {
    return quantile(p, shape1, shape2);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
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

test probability {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0       , probability(0  , 3, 5), eps);
    try expectApproxEqRel(0.148032, probability(0.2, 3, 5), eps);
    try expectApproxEqRel(0.995328, probability(0.8, 3, 5), eps);
    try expectApproxEqRel(1       , probability(1  , 3, 5), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 3, 5));
    try expectEqual(0, survival( inf, 3, 5));

    try expectApproxEqRel(1       , survival(0  , 3, 5), eps);
    try expectApproxEqRel(0.851968, survival(0.2, 3, 5), eps);
    try expectApproxEqRel(0.004672, survival(0.8, 3, 5), eps);
    try expectApproxEqRel(0       , survival(1  , 3, 5), eps);
}

test quantile {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.2283264643498391, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.3205858305642004, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(0.4092151219095550, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(0.5167577700975785, quantile(0.8, 3, 5), eps);
    try expectApproxEqRel(1                 , quantile(1  , 3, 5), eps);
}
