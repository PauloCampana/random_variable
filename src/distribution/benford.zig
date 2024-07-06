//! Support: {1,2,⋯,b - 1}
//!
//! Parameters:
//! - b: `base` ∈ {2,3,4,⋯}

const std = @import("std");
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// p(x) = log_b(1 + 1 / x)
pub fn density(x: f64, base: u64) f64 {
    assert(base >= 2);
    assert(!isNan(x));
    const b: f64 = @floatFromInt(base);
    if (x < 1 or x > b - 1 or x != @round(x)) {
        return 0;
    }
    return std.math.log1p(1 / x) / @log(b);
}

/// F(q) = log_b(1 + ⌊q⌋)
pub fn probability(q: f64, base: u64) f64 {
    assert(base >= 2);
    assert(!isNan(q));
    const b: f64 = @floatFromInt(base);
    if (q < 1) {
        return 0;
    }
    if (q >= b - 1) {
        return 1;
    }
    return std.math.log1p(@floor(q)) / @log(b);
}

/// S(t) = log_b(b / (1 + ⌊t⌋))
pub fn survival(t: f64, base: u64) f64 {
    assert(base >= 2);
    assert(!isNan(t));
    const b: f64 = @floatFromInt(base);
    if (t < 1) {
        return 1;
    }
    if (t >= b - 1) {
        return 0;
    }
    return @log(b / (1 + @floor(t))) / @log(b);
}

/// Q(p) = ⌈b^p⌉ - 1
pub fn quantile(p: f64, base: u64) f64 {
    assert(base >= 2);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return 1;
    }
    const bp = std.math.pow(f64, @floatFromInt(base), p);
    return @ceil(bp) - 1;
}

pub fn random(generator: std.Random, base: u64) f64 {
    assert(base >= 2);
    const uni = generator.float(f64);
    if (base == 2) {
        return 1;
    }
    const bp = std.math.pow(f64, @floatFromInt(base), uni);
    return @ceil(bp) - 1;
}

pub fn fill(buffer: []f64, generator: std.Random, base: u64) void {
    assert(base >= 2);
    if (base == 2) {
        return @memset(buffer, 1);
    }
    for (buffer) |*x| {
        const uni = generator.float(f64);
        const bp = std.math.pow(f64, @floatFromInt(base), uni);
        x.* = @ceil(bp) - 1;
    }
}

export fn rv_benford_density(x: f64, base: u64) f64 {
    return density(x, base);
}
export fn rv_benford_probability(q: f64, base: u64) f64 {
    return probability(q, base);
}
export fn rv_benford_survival(t: f64, base: u64) f64 {
    return survival(t, base);
}
export fn rv_benford_quantile(p: f64, base: u64) f64 {
    return quantile(p, base);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 10));
    try expectEqual(0, density( inf, 10));

    try expectEqual(0, density(0.9, 2));
    try expectEqual(1, density(1  , 2));
    try expectEqual(0, density(1.1, 2));

    try expectApproxEqRel(0                 , density(0.9, 10), eps);
    try expectApproxEqRel(0.3010299956639811, density(1  , 10), eps);
    try expectApproxEqRel(0                 , density(1.1, 10), eps);
    try expectApproxEqRel(0                 , density(1.9, 10), eps);
    try expectApproxEqRel(0.1760912590556812, density(2  , 10), eps);
    try expectApproxEqRel(0                 , density(2.1, 10), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 10));
    try expectEqual(1, probability( inf, 10));

    try expectEqual(0, probability(0.9, 2));
    try expectEqual(1, probability(1  , 2));
    try expectEqual(1, probability(1.1, 2));

    try expectApproxEqRel(0                 , probability(0.9, 10), eps);
    try expectApproxEqRel(0.3010299956639812, probability(1  , 10), eps);
    try expectApproxEqRel(0.3010299956639812, probability(1.1, 10), eps);
    try expectApproxEqRel(0.3010299956639812, probability(1.9, 10), eps);
    try expectApproxEqRel(0.4771212547196624, probability(2  , 10), eps);
    try expectApproxEqRel(0.4771212547196624, probability(2.1, 10), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 10));
    try expectEqual(0, survival( inf, 10));

    try expectEqual(1, survival(0.9, 2));
    try expectEqual(0, survival(1  , 2));
    try expectEqual(0, survival(1.1, 2));

    try expectApproxEqRel(1                 , survival(0.9, 10), eps);
    try expectApproxEqRel(0.6989700043360188, survival(1  , 10), eps);
    try expectApproxEqRel(0.6989700043360188, survival(1.1, 10), eps);
    try expectApproxEqRel(0.6989700043360188, survival(1.9, 10), eps);
    try expectApproxEqRel(0.5228787452803375, survival(2  , 10), eps);
    try expectApproxEqRel(0.5228787452803375, survival(2.1, 10), eps);
}

test quantile {
    try expectEqual(1, quantile(0  , 2));
    try expectEqual(1, quantile(0.5, 2));
    try expectEqual(1, quantile(1  , 2));

    try expectEqual(1, quantile(0                 , 10));
    try expectEqual(1, quantile(0.3010299956639811, 10));
    try expectEqual(1, quantile(0.3010299956639812, 10));
    try expectEqual(2, quantile(0.3010299956639813, 10));
    try expectEqual(2, quantile(0.4771212547196623, 10));
    try expectEqual(2, quantile(0.4771212547196624, 10));
    try expectEqual(3, quantile(0.4771212547196625, 10));
    try expectEqual(9, quantile(1                 , 10));
}
