//! Support: {max(0, n + K - N),1,⋯,min(n, K)}
//!
//! Parameters:
//! - N: `total` ∈ {0,1,2,⋯}
//! - K: `good`  ∈ {0,1,⋯,N}
//! - n: `tries` ∈ {0,1,⋯,N}

const std = @import("std");
const special = @import("../special.zig");
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// p(x) = (K x) (N - K n - x) / (N n)
pub fn density(x: f64, total: u64, good: u64, tries: u64) f64 {
    assert(good <= total and tries <= total);
    assert(!isNan(x));
    const low: f64 = @floatFromInt(lower(total, good, tries));
    const upp: f64 = @floatFromInt(upper(total, good, tries));
    if (x < low or x > upp or x != @round(x)) {
        return 0;
    }
    const f_total: f64 = @floatFromInt(total);
    const f_good: f64 = @floatFromInt(good);
    const f_tries: f64 = @floatFromInt(tries);
    const num1 = special.lbinomial(f_good, x);
    const num2 = special.lbinomial(f_total - f_good, f_tries - x);
    const den = special.lbinomial(f_total, f_tries);
    return @exp(num1 + num2 - den);
}

/// No closed form
pub fn probability(q: f64, total: u64, good: u64, tries: u64) f64 {
    assert(good <= total and tries <= total);
    assert(!isNan(q));
    const low: f64 = @floatFromInt(lower(total, good, tries));
    const upp: f64 = @floatFromInt(upper(total, good, tries));
    if (q < low) {
        return 0;
    }
    if (q >= upp) {
        return 1;
    }
    var hypr = lower(total, good, tries);
    var mass = density(low, total, good, tries);
    var cumu = mass;
    for (0..@intFromFloat(q)) |_| {
        const num: f64 = @floatFromInt((good - hypr) * (tries - hypr));
        hypr += 1;
        const den: f64 = @floatFromInt(hypr * (hypr + total - good - tries));
        mass *= num / den;
        cumu += mass;
    }
    return cumu;
}

/// No closed form
pub fn quantile(p: f64, total: u64, good: u64, tries: u64) f64 {
    assert(good <= total and tries <= total);
    assert(0 <= p and p <= 1);
    const low = lower(total, good, tries);
    const upp = upper(total, good, tries);
    if (p == 0) {
        return @floatFromInt(low);
    }
    if (p == 1) {
        return @floatFromInt(upp);
    }
    if (low == upp) {
        return @floatFromInt(low);
    }
    const initial_mass = density(@floatFromInt(low), total, good, tries);
    return linearSearch(p, total, good, tries, low, initial_mass);
}

pub fn random(generator: std.Random, total: u64, good: u64, tries: u64) f64 {
    assert(good <= total and tries <= total);
    const low = lower(total, good, tries);
    const upp = upper(total, good, tries);
    if (low == upp) {
        return @floatFromInt(low);
    }
    const initial_mass = density(@floatFromInt(low), total, good, tries);
    const uni = generator.float(f64);
    return linearSearch(uni, total, good, tries, low, initial_mass);
}

pub fn fill(buffer: []f64, generator: std.Random, total: u64, good: u64, tries: u64) void {
    assert(good <= total and tries <= total);
    const low = lower(total, good, tries);
    const upp = upper(total, good, tries);
    if (low == upp) {
        return @memset(buffer, @floatFromInt(low));
    }
    const initial_mass = density(@floatFromInt(low), total, good, tries);
    for (buffer) |*x| {
        const uni = generator.float(f64);
        x.* = linearSearch(uni, total, good, tries, low, initial_mass);
    }
}

fn linearSearch(p: f64, total: u64, good: u64, tries: u64, initial: u64, initial_mass: f64) f64 {
    var hypr = initial;
    var mass = initial_mass;
    var cumu = mass;
    while (cumu <= p) {
        const num: f64 = @floatFromInt((good - hypr) * (tries - hypr));
        hypr += 1;
        const den: f64 = @floatFromInt(hypr * (hypr + total - good - tries));
        mass *= num / den;
        cumu += mass;
    }
    return @floatFromInt(hypr);
}

fn lower(total: u64, good: u64, tries: u64) u64 {
    return tries + good -| total;
}

fn upper(_: u64, good: u64, tries: u64) u64 {
    return @min(good, tries);
}

export fn rv_hypergeometric_density(x: f64, total: u64, good: u64, tries: u64) f64 {
    return density(x, total, good, tries);
}
export fn rv_hypergeometric_probability(q: f64, total: u64, good: u64, tries: u64) f64 {
    return probability(q, total, good, tries);
}
export fn rv_hypergeometric_quantile(p: f64, total: u64, good: u64, tries: u64) f64 {
    return quantile(p, total, good, tries);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 10, 2, 5));
    try expectEqual(0, density( inf, 10, 2, 5));

    try expectEqual(1, density( 0, 10,  2,  0));
    try expectEqual(0, density( 1, 10,  2,  0));
    try expectEqual(1, density( 0, 10,  0,  5));
    try expectEqual(0, density( 1, 10,  0,  5));
    try expectEqual(0, density( 4, 10, 10,  5));
    try expectEqual(1, density( 5, 10, 10,  5));
    try expectEqual(0, density( 6, 10, 10,  5));
    try expectEqual(0, density( 1, 10,  2, 10));
    try expectEqual(1, density( 2, 10,  2, 10));
    try expectEqual(0, density( 3, 10,  2, 10));
    try expectEqual(0, density( 9, 10, 10, 10));
    try expectEqual(1, density(10, 10, 10, 10));
    try expectEqual(0, density(11, 10, 10, 10));

    try expectApproxEqRel(0                 , density(-0.1, 10, 2, 5), eps);
    try expectApproxEqRel(0.2222222222222222, density( 0  , 10, 2, 5), eps);
    try expectApproxEqRel(0                 , density( 0.1, 10, 2, 5), eps);
    try expectApproxEqRel(0                 , density( 0.9, 10, 2, 5), eps);
    try expectApproxEqRel(0.5555555555555556, density( 1  , 10, 2, 5), eps);
    try expectApproxEqRel(0                 , density( 1.1, 10, 2, 5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 10, 2, 5));
    try expectEqual(1, probability( inf, 10, 2, 5));

    try expectEqual(1, probability( 0, 10,  2, 0 ));
    try expectEqual(1, probability( 1, 10,  2, 0 ));
    try expectEqual(1, probability( 0, 10,  0, 5 ));
    try expectEqual(1, probability( 1, 10,  0, 5 ));
    try expectEqual(0, probability( 4, 10, 10, 5 ));
    try expectEqual(1, probability( 5, 10, 10, 5 ));
    try expectEqual(1, probability( 6, 10, 10, 5 ));
    try expectEqual(0, probability( 1, 10,  2, 10));
    try expectEqual(1, probability( 2, 10,  2, 10));
    try expectEqual(1, probability( 3, 10,  2, 10));
    try expectEqual(0, probability( 9, 10, 10, 10));
    try expectEqual(1, probability(10, 10, 10, 10));
    try expectEqual(1, probability(11, 10, 10, 10));

    try expectApproxEqRel(0                 , probability(-0.1, 10, 2, 5), eps);
    try expectApproxEqRel(0.2222222222222222, probability( 0  , 10, 2, 5), eps);
    try expectApproxEqRel(0.2222222222222222, probability( 0.1, 10, 2, 5), eps);
    try expectApproxEqRel(0.2222222222222222, probability( 0.9, 10, 2, 5), eps);
    try expectApproxEqRel(0.7777777777777778, probability( 1  , 10, 2, 5), eps);
    try expectApproxEqRel(0.7777777777777778, probability( 1.1, 10, 2, 5), eps);
}

test quantile {
    try expectEqual( 0, quantile(0  , 10,  2, 0 ));
    try expectEqual( 0, quantile(0.5, 10,  2, 0 ));
    try expectEqual( 0, quantile(1  , 10,  2, 0 ));
    try expectEqual( 0, quantile(0  , 10,  0, 5 ));
    try expectEqual( 0, quantile(0.5, 10,  0, 5 ));
    try expectEqual( 0, quantile(1  , 10,  0, 5 ));
    try expectEqual( 5, quantile(0  , 10, 10, 5 ));
    try expectEqual( 5, quantile(0.5, 10, 10, 5 ));
    try expectEqual( 5, quantile(1  , 10, 10, 5 ));
    try expectEqual( 2, quantile(0  , 10,  2, 10));
    try expectEqual( 2, quantile(0.5, 10,  2, 10));
    try expectEqual( 2, quantile(1  , 10,  2, 10));
    try expectEqual(10, quantile(0  , 10, 10, 10));
    try expectEqual(10, quantile(0.5, 10, 10, 10));
    try expectEqual(10, quantile(1  , 10, 10, 10));

    try expectEqual(0, quantile(0                 , 10, 2, 5));
    try expectEqual(0, quantile(0.2222222222222221, 10, 2, 5));
    try expectEqual(0, quantile(0.2222222222222222, 10, 2, 5));
    try expectEqual(1, quantile(0.2222222222222223, 10, 2, 5));
    try expectEqual(1, quantile(0.7777777777777777, 10, 2, 5));
    try expectEqual(1, quantile(0.7777777777777778, 10, 2, 5));
    try expectEqual(2, quantile(0.7777777777777780, 10, 2, 5));
    try expectEqual(2, quantile(1                 , 10, 2, 5));
}
