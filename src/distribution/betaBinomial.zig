//! Support: X ∈ {0,1,⋯,n}
//!
//! Parameters:
//! - n: `size`   ∈ {0,1,2,⋯}
//! - α: `shape1` ∈ (0,∞)
//! - β: `shape2` ∈ (0,∞)

const std = @import("std");
const special = @import("../special.zig");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = true;

/// p(x) = (n x) beta(x + α, n - x + β) / beta(α, β).
pub fn density(x: f64, size: u64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(x));
    const n = @as(f64, @floatFromInt(size));
    if (x < 0 or x > n or x != @round(x)) {
        return 0;
    }
    const binom = special.lbinomial(n, x);
    const beta1 = special.lbeta(x + shape1, n - x + shape2);
    const beta2 = special.lbeta(shape1, shape2);
    return @exp(binom + beta1 - beta2);
}

/// No closed form.
pub fn probability(q: f64, size: u64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    const n = @as(f64, @floatFromInt(size));
    if (q >= n) {
        return 1;
    }
    const mass_num = special.lbeta(shape1, shape2 + n);
    const mass_den = special.lbeta(shape1, shape2);
    var mass = @exp(mass_num - mass_den);
    var cumu = mass;
    var bbin: f64 = 0;
    for (0..@intFromFloat(q)) |_| {
        const num = (n - bbin) * (shape1 + bbin);
        bbin += 1;
        const den = bbin * (shape2 + n - bbin);
        mass *= num / den;
        cumu += mass;
    }
    return cumu;
}

/// No closed form.
pub fn quantile(p: f64, size: u64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(0 <= p and p <= 1);
    const n = @as(f64, @floatFromInt(size));
    if (p == 0 or p == 1 or size == 0) {
        return n * p;
    }
    const mass_num = special.lbeta(shape1, shape2 + n);
    const mass_den = special.lbeta(shape1, shape2);
    const initial_mass = @exp(mass_num - mass_den);
    return linearSearch(p, n, shape1, shape2, initial_mass);
}

pub fn random(generator: std.Random, size: u64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    if (size == 0) {
        return 0;
    }
    const n = @as(f64, @floatFromInt(size));
    const mass_num = special.lbeta(shape1, shape2 + n);
    const mass_den = special.lbeta(shape1, shape2);
    const initial_mass = @exp(mass_num - mass_den);
    const uni = generator.float(f64);
    return linearSearch(uni, n, shape1, shape2, initial_mass);
}

pub fn fill(buffer: []f64, generator: std.Random, size: u64, shape1: f64, shape2: f64) []f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    if (size == 0) {
        @memset(buffer, 0);
        return buffer;
    }
    const n = @as(f64, @floatFromInt(size));
    const mass_num = special.lbeta(shape1, shape2 + n);
    const mass_den = special.lbeta(shape1, shape2);
    const initial_mass = @exp(mass_num - mass_den);
    for (buffer) |*x| {
        const uni = generator.float(f64);
        x.* = linearSearch(uni, n, shape1, shape2, initial_mass);
    }
    return buffer;
}

fn linearSearch(p: f64, n: f64, shape1: f64, shape2: f64, initial_mass: f64) f64 {
    var bbin: f64 = 0;
    var mass = initial_mass;
    var cumu = mass;
    while (cumu <= p) {
        const num = (n - bbin) * (shape1 + bbin);
        bbin += 1;
        const den = bbin * (shape2 + n - bbin);
        mass *= num / den;
        cumu += mass;
    }
    return bbin;
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 30 * std.math.floatEps(f64); // 6.66 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 10, 3, 5));
    try expectEqual(0, density( inf, 10, 3, 5));

    try expectEqual(1, density(0, 0, 3, 5));
    try expectEqual(0, density(1, 0, 3, 5));

    try expectApproxEqRel(0                  , density(-0.1, 10, 3, 5), eps);
    try expectApproxEqRel(0.05147058823529412, density( 0  , 10, 3, 5), eps);
    try expectApproxEqRel(0                  , density( 0.1, 10, 3, 5), eps);
    try expectApproxEqRel(0                  , density( 0.9, 10, 3, 5), eps);
    try expectApproxEqRel(0.11029411764705882, density( 1  , 10, 3, 5), eps);
    try expectApproxEqRel(0                  , density( 1.1, 10, 3, 5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 10, 3, 5));
    try expectEqual(1, probability( inf, 10, 3, 5));

    try expectEqual(1, probability(0, 0, 3, 5));
    try expectEqual(1, probability(1, 0, 3, 5));

    try expectApproxEqRel(0                  , probability(-0.1, 10, 3, 5), eps);
    try expectApproxEqRel(0.05147058823529412, probability( 0  , 10, 3, 5), eps);
    try expectApproxEqRel(0.05147058823529412, probability( 0.1, 10, 3, 5), eps);
    try expectApproxEqRel(0.05147058823529412, probability( 0.9, 10, 3, 5), eps);
    try expectApproxEqRel(0.16176470588235294, probability( 1  , 10, 3, 5), eps);
    try expectApproxEqRel(0.16176470588235294, probability( 1.1, 10, 3, 5), eps);
}

test quantile {
    try expectEqual(0, quantile(0  , 0, 3, 5));
    try expectEqual(0, quantile(0.5, 0, 3, 5));
    try expectEqual(0, quantile(1  , 0, 3, 5));

    try expectEqual( 0, quantile(0                , 10, 3, 5));
    try expectEqual( 0, quantile(0.051470588235292, 10, 3, 5));
    try expectEqual( 0, quantile(0.051470588235293, 10, 3, 5));
    try expectEqual( 1, quantile(0.051470588235294, 10, 3, 5));
    try expectEqual( 1, quantile(0.161764705882351, 10, 3, 5));
    try expectEqual( 1, quantile(0.161764705882352, 10, 3, 5));
    try expectEqual( 2, quantile(0.161764705882353, 10, 3, 5));
    try expectEqual(10, quantile(1                , 10, 3, 5));
}
