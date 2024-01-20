//! Support: X ∈ [0,∞)
//!
//! Parameters:
//! - ν: `df` ∈ (0,∞)

const std = @import("std");
const gamma = @import("gamma.zig");
const incompleteGamma = @import("../thirdyparty/prob.zig").incompleteGamma;
const inverseComplementedIncompleteGamma = @import("../thirdyparty/prob.zig").inverseComplementedIncompleteGamma;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = false;
pub const parameters = 1;

/// f(x) = x^(ν - 1) exp(-x^2 / 2) / (2^(ν / 2 - 1) gamma(ν / 2)).
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

/// No closed form.
pub fn probability(q: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    return incompleteGamma(0.5 * df, 0.5 * q * q);
}

/// No closed form.
pub fn quantile(p: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return 0;
    }
    if (p == 1) {
        return inf;
    }
    const q = inverseComplementedIncompleteGamma(0.5 * df, 1 - p);
    return @sqrt(2 * q);
}

/// Uses the relation to Gamma distribution.
pub const random = struct {
    pub fn single(generator: std.rand.Random, df: f64) f64 {
        assert(isFinite(df));
        assert(df > 0);
        const chisq = gamma.random.single(generator, 0.5 * df, 0.5);
        return @sqrt(chisq);
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, df: f64) []f64 {
        assert(isFinite(df));
        assert(df > 0);
        const hdf = 0.5 * df;
        for (buffer) |*x| {
            const chisq = gamma.random.single(generator, hdf, 0.5);
            x.* = @sqrt(chisq);
        }
        return buffer;
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "chi.density" {
    try expectEqual(0, density(-inf, 3));
    try expectEqual(0, density( inf, 3));

    try expectEqual(inf                     , density(0, 0.9));
    try expectEqual(@sqrt(2.0 / std.math.pi), density(0, 1  ));
    try expectEqual(0                       , density(0, 1.1));

    try expectApproxEqRel(0                 , density(0, 3), eps);
    try expectApproxEqRel(0.4839414490382866, density(1, 3), eps);
    try expectApproxEqRel(0.4319277321055044, density(2, 3), eps);
}

test "chi.probability" {
    try expectEqual(0, probability(-inf, 3));
    try expectEqual(1, probability( inf, 3));

    try expectApproxEqRel(0                 , probability(0, 3), eps);
    try expectApproxEqRel(0.1987480430987991, probability(1, 3), eps);
    try expectApproxEqRel(0.7385358700508893, probability(2, 3), eps);
}

test "chi.quantile" {
    try expectApproxEqRel(0                , quantile(0  , 3), eps);
    try expectApproxEqRel(1.002583668853801, quantile(0.2, 3), eps);
    try expectApproxEqRel(1.367175337470916, quantile(0.4, 3), eps);
    try expectApproxEqRel(1.716439941594797, quantile(0.6, 3), eps);
    try expectApproxEqRel(2.154443704552858, quantile(0.8, 3), eps);
    try expectEqual      (inf              , quantile(1  , 3)     );
}

test "chi.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.534269c2e0eadp0, random.single(gen, 3), eps);
    try expectApproxEqRel(0x1.742316623cc60p0, random.single(gen, 3), eps);
    try expectApproxEqRel(0x1.85d3662a50bb2p0, random.single(gen, 3), eps);
}
