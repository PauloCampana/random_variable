//! Support: {1,2,3,⋯}
//!
//! Parameters:
//! - p: `prob` ∈ (0,1)

const std = @import("std");
const assert = @import("../assert.zig");
const inf = std.math.inf(f64);

/// p(x) = p^x / (-ln(1 - p) x)
pub fn density(x: f64, prob: f64) callconv(.c) f64 {
    assert.logarithmic(prob);
    assert.real(x);

    if (x < 1 or x != @round(x)) {
        return 0;
    }
    const pow = std.math.pow(f64, prob, x);
    const ln = -std.math.log1p(-prob);
    return pow / (x * ln);
}

/// No closed form
pub fn probability(q: f64, prob: f64) callconv(.c) f64 {
    assert.logarithmic(prob);
    assert.real(q);

    if (q < 1) {
        return 0;
    }
    if (q == inf) {
        return 1;
    }
    var loga: f64 = 1;
    var mass = prob / -std.math.log1p(-prob);
    var cumu = mass;
    for (1..@intFromFloat(q)) |_| {
        const num = prob * loga;
        loga += 1;
        mass *= num / loga;
        cumu += mass;
    }
    return cumu;
}

/// No closed form
pub fn survival(t: f64, prob: f64) callconv(.c) f64 {
    return 1 - probability(t, prob);
}

/// No closed form
pub fn quantile(p: f64, prob: f64) callconv(.c) f64 {
    assert.logarithmic(prob);
    assert.probability(p);

    if (p == 0) {
        return 1;
    }
    if (p == 1) {
        return inf;
    }
    const initial_mass = prob / -std.math.log1p(-prob);
    return linearSearch(p, prob, initial_mass);
}

pub fn random(generator: std.Random, prob: f64) f64 {
    assert.logarithmic(prob);

    const initial_mass = prob / -std.math.log1p(-prob);
    const uni = generator.float(f64);
    return linearSearch(uni, prob, initial_mass);
}

pub fn fill(buffer: []f64, generator: std.Random, prob: f64) void {
    assert.logarithmic(prob);

    const initial_mass = prob / -std.math.log1p(-prob);
    for (buffer) |*x| {
        const uni = generator.float(f64);
        x.* = linearSearch(uni, prob, initial_mass);
    }
}

fn linearSearch(p: f64, prob: f64, initial_mass: f64) f64 {
    var loga: f64 = 1;
    var mass = initial_mass;
    var cumu = mass;
    while (cumu <= p) {
        const num = prob * loga;
        loga += 1;
        mass *= num / loga;
        cumu += mass;
    }
    return loga;
}

comptime {
    @export(&density, .{ .name = "rv_logarithmic_density" });
    @export(&probability, .{ .name = "rv_logarithmic_probability" });
    @export(&survival, .{ .name = "rv_logarithmic_survival" });
    @export(&quantile, .{ .name = "rv_logarithmic_quantile" });
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 0.2));
    try expectEqual(0, density( inf, 0.2));

    try expectApproxEqRel(0                 , density(0.9, 0.2), eps);
    try expectApproxEqRel(0.8962840235449099, density(1  , 0.2), eps);
    try expectApproxEqRel(0                 , density(1.1, 0.2), eps);
    try expectApproxEqRel(0                 , density(1.9, 0.2), eps);
    try expectApproxEqRel(0.0896284023544909, density(2  , 0.2), eps);
    try expectApproxEqRel(0                 , density(2.1, 0.2), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 0.2));
    try expectEqual(1, probability( inf, 0.2));

    try expectApproxEqRel(0                 , probability(0.9, 0.2), eps);
    try expectApproxEqRel(0.8962840235449099, probability(1  , 0.2), eps);
    try expectApproxEqRel(0.8962840235449099, probability(1.1, 0.2), eps);
    try expectApproxEqRel(0.8962840235449099, probability(1.9, 0.2), eps);
    try expectApproxEqRel(0.9859124258994009, probability(2  , 0.2), eps);
    try expectApproxEqRel(0.9859124258994009, probability(2.1, 0.2), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 0.2));
    try expectEqual(0, survival( inf, 0.2));

    try expectApproxEqRel(1                  , survival(0.9, 0.2), eps);
    try expectApproxEqRel(0.10371597645509004, survival(1  , 0.2), eps);
    try expectApproxEqRel(0.10371597645509004, survival(1.1, 0.2), eps);
    try expectApproxEqRel(0.10371597645509004, survival(1.9, 0.2), eps);
    try expectApproxEqRel(0.01408757410059904, survival(2  , 0.2), eps);
    try expectApproxEqRel(0.01408757410059904, survival(2.1, 0.2), eps);
}

test quantile {
    try expectEqual(  1, quantile(0                 , 0.2));
    try expectEqual(  1, quantile(0.8962840235449098, 0.2));
    try expectEqual(  1, quantile(0.8962840235449099, 0.2));
    try expectEqual(  2, quantile(0.8962840235449100, 0.2));
    try expectEqual(  2, quantile(0.9859124258994008, 0.2));
    try expectEqual(  2, quantile(0.9859124258994009, 0.2));
    try expectEqual(  3, quantile(0.9859124258994010, 0.2));
    try expectEqual(inf, quantile(1                 , 0.2));
}
