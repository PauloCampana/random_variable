//! Support: [0,∞)
//!
//! Parameters:
//! - α: `shape1` ∈ (0,∞)
//! - β: `shape2` ∈ (0,∞)

const std = @import("std");
const assert = @import("../assert.zig");
const special = @import("../special.zig");
const gamma = @import("gamma.zig");
const inf = std.math.inf(f64);

/// f(x) = x^(α - 1) (1 + x)^(-α - β) / beta(α, β)
pub fn density(x: f64, shape1: f64, shape2: f64) callconv(.c) f64 {
    assert.beta_prime(shape1, shape2);
    assert.real(x);

    if (x < 0 or x == inf) {
        return 0;
    }
    if (x == 0) {
        if (shape1 == 1) {
            return shape2;
        }
        return if (shape1 < 1) inf else 0;
    }
    const num = (shape1 - 1) * @log(x) - (shape1 + shape2) * std.math.log1p(x);
    const den = special.lbeta(shape1, shape2);
    return @exp(num - den);
}

/// No closed form
pub fn probability(q: f64, shape1: f64, shape2: f64) callconv(.c) f64 {
    assert.beta_prime(shape1, shape2);
    assert.real(q);

    if (q <= 0) {
        return 0;
    }
    if (q == inf) {
        return 1;
    }
    const z = q / (1 + q);
    return special.beta.probability(z, shape1, shape2);
}

/// No closed form
pub fn survival(t: f64, shape1: f64, shape2: f64) callconv(.c) f64 {
    assert.beta_prime(shape1, shape2);
    assert.real(t);

    if (t <= 0) {
        return 1;
    }
    const z = 1 / (1 + t);
    return special.beta.probability(z, shape2, shape1);
}

/// No closed form
pub fn quantile(p: f64, shape1: f64, shape2: f64) callconv(.c) f64 {
    assert.beta_prime(shape1, shape2);
    assert.probability(p);

    const q = special.beta.quantile(p, shape1, shape2);
    return q / (1 - q);
}

pub fn random(generator: std.Random, shape1: f64, shape2: f64) f64 {
    assert.beta_prime(shape1, shape2);

    if (shape1 == 1) {
        const uni = generator.float(f64);
        const b = std.math.pow(f64, uni, 1 / shape2);
        return (1 - b) / b;
    }
    if (shape2 == 1) {
        const uni = generator.float(f64);
        const a = std.math.pow(f64, uni, 1 / shape1);
        return a / (1 - a);
    }
    if (shape1 < 1 and shape2 < 1) {
        return rejection(generator, 1 / shape1, 1 / shape2);
    }
    const gam1 = gamma.random(generator, shape1, 1);
    const gam2 = gamma.random(generator, shape2, 1);
    return gam1 / gam2;
}

pub fn fill(buffer: []f64, generator: std.Random, shape1: f64, shape2: f64) void {
    assert.beta_prime(shape1, shape2);

    const invshape1 = 1 / shape1;
    const invshape2 = 1 / shape2;
    if (shape1 == 1) {
        for (buffer) |*x| {
            const uni = generator.float(f64);
            const b = std.math.pow(f64, uni, invshape2);
            x.* = (1 - b) / b;
        }
        return;
    }
    if (shape2 == 1) {
        for (buffer) |*x| {
            const uni = generator.float(f64);
            const a = std.math.pow(f64, uni, invshape1);
            x.* = a / (1 - a);
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
        x.* = gam1 / gam2;
    }
}

fn rejection(generator: std.Random, invshape1: f64, invshape2: f64) f64 {
    while (true) {
        const uni1 = generator.float(f64);
        const uni2 = generator.float(f64);
        const x = std.math.pow(f64, uni1, invshape1);
        const y = std.math.pow(f64, uni2, invshape2);
        if (x + y <= 1) {
            return x / y;
        }
    }
}

comptime {
    @export(&density, .{ .name = "rv_beta_prime_density" });
    @export(&probability, .{ .name = "rv_beta_prime_probability" });
    @export(&survival, .{ .name = "rv_beta_prime_survival" });
    @export(&quantile, .{ .name = "rv_beta_prime_quantile" });
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectEqual(inf, density(0, 0.9, 5));
    try expectEqual(  5, density(0, 1  , 5));
    try expectEqual(  0, density(0, 1.1, 5));

    try expectApproxEqRel(0                  , density(0, 3, 5), eps);
    try expectApproxEqRel(0.41015625         , density(1, 3, 5), eps);
    try expectApproxEqRel(0.06401463191586648, density(2, 3, 5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                 , probability(0, 3, 5), eps);
    try expectApproxEqRel(0.7734375         , probability(1, 3, 5), eps);
    try expectApproxEqRel(0.9547325102880658, probability(2, 3, 5), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 3, 5));
    try expectEqual(0, survival( inf, 3, 5));

    try expectApproxEqRel(1                  , survival(0, 3, 5), eps);
    try expectApproxEqRel(0.2265625          , survival(1, 3, 5), eps);
    try expectApproxEqRel(0.04526748971193415, survival(2, 3, 5), eps);
}

test quantile {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.2958847929875766, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.4718562623302689, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(0.6926635008537015, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(1.0693555697769304, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf               , quantile(1  , 3, 5)     );
}
