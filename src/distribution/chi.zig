//! Support: [0,∞)
//!
//! Parameters:
//! - ν: `df` ∈ (0,∞)

const std = @import("std");
const gamma = @import("gamma.zig");
const special = @import("../special.zig");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = x^(ν - 1) exp(-x^2 / 2) / (2^(ν / 2 - 1) gamma(ν / 2))
pub fn density(x: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(!isNan(x));
    if (x < 0 or x == inf) {
        return 0;
    }
    if (x == 0) {
        if (df == 1) {
            return comptime @sqrt(2.0 / std.math.pi);
        }
        return if (df < 1) inf else 0;
    }
    const hdf = 0.5 * df;
    const num = (df - 1) * @log(x) - 0.5 * x * x;
    const den = (hdf - 1) * std.math.ln2 + std.math.lgamma(f64, hdf);
    return @exp(num - den);
}

/// No closed form
pub fn probability(q: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    return special.gamma.probability(0.5 * q * q, 0.5 * df);
}

/// No closed form
pub fn survival(t: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(!isNan(t));
    if (t <= 0) {
        return 1;
    }
    return special.gamma.survival(0.5 * t * t, 0.5 * df);
}

/// No closed form
pub fn quantile(p: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(0 <= p and p <= 1);
    const q = special.gamma.quantile(p, 0.5 * df);
    return @sqrt(2 * q);
}

pub fn random(generator: std.Random, df: f64) f64 {
    const chisq = gamma.random(generator, 0.5 * df, 2);
    return @sqrt(chisq);
}

pub fn fill(buffer: []f64, generator: std.Random, df: f64) void {
    const hdf = 0.5 * df;
    for (buffer) |*x| {
        const chisq = gamma.random(generator, hdf, 2);
        x.* = @sqrt(chisq);
    }
}

export fn rv_chi_density(x: f64, df: f64) f64 {
    return density(x, df);
}
export fn rv_chi_probability(q: f64, df: f64) f64 {
    return probability(q, df);
}
export fn rv_chi_survival(t: f64, df: f64) f64 {
    return survival(t, df);
}
export fn rv_chi_quantile(p: f64, df: f64) f64 {
    return quantile(p, df);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3));
    try expectEqual(0, density( inf, 3));

    try expectEqual(inf                     , density(0, 0.9));
    try expectEqual(@sqrt(2.0 / std.math.pi), density(0, 1  ));
    try expectEqual(0                       , density(0, 1.1));

    try expectApproxEqRel(0                 , density(0, 3), eps);
    try expectApproxEqRel(0.4839414490382866, density(1, 3), eps);
    try expectApproxEqRel(0.4319277321055044, density(2, 3), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3));
    try expectEqual(1, probability( inf, 3));

    try expectApproxEqRel(0                 , probability(0, 3), eps);
    try expectApproxEqRel(0.1987480430987991, probability(1, 3), eps);
    try expectApproxEqRel(0.7385358700508893, probability(2, 3), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 3));
    try expectEqual(0, survival( inf, 3));

    try expectApproxEqRel(1                 , survival(0, 3), eps);
    try expectApproxEqRel(0.8012519569012008, survival(1, 3), eps);
    try expectApproxEqRel(0.2614641299491106, survival(2, 3), eps);
}

test quantile {
    try expectApproxEqRel(0                , quantile(0  , 3), eps);
    try expectApproxEqRel(1.002583668853801, quantile(0.2, 3), eps);
    try expectApproxEqRel(1.367175337470916, quantile(0.4, 3), eps);
    try expectApproxEqRel(1.716439941594797, quantile(0.6, 3), eps);
    try expectApproxEqRel(2.154443704552858, quantile(0.8, 3), eps);
    try expectEqual      (inf              , quantile(1  , 3)     );
}
