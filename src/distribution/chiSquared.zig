//! Parameters:
//! - ν: `df` ∈ (0,∞)

const std = @import("std");
const gamma = @import("gamma.zig");
const inf = std.math.inf(f64);

pub const parameters = 1;
pub const support = [2]f64 {0, inf};

/// f(x) = 0.5 / gamma(ν / 2) (x / 2)^(ν / 2 - 1) exp(-x / 2).
pub fn density(x: f64, df: f64) f64 {
    return gamma.density(x, 0.5 * df, 0.5);
}

/// No closed form.
pub fn probability(q: f64, df: f64) f64 {
    return gamma.probability(q, 0.5 * df, 0.5);
}

/// No closed form.
pub fn quantile(p: f64, df: f64) f64 {
    return gamma.quantile(p, 0.5 * df, 0.5);
}

/// Uses the relation to Gamma distribution.
pub const random = struct {
    fn implementation(generator: std.rand.Random, df1: f64) f64 {
        return gamma.random.implementation(generator, 0.5 * df1, 0.5);
    }

    pub fn single(generator: std.rand.Random, df: f64) f64 {
        return gamma.random.single(generator, 0.5 * df, 0.5);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, df: f64) []f64 {
        return gamma.random.buffer(buf, generator, 0.5 * df, 0.5);
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, df: f64) ![]f64 {
        return gamma.random.alloc(allocator, generator, n, 0.5 * df, 0.5);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "chiSquared.density" {
    try expectEqual(0, density(-inf, 3));
    try expectEqual(0, density( inf, 3));

    try expectEqual(inf, density(0, 1.8));
    try expectEqual(0.5, density(0, 2  ));
    try expectEqual(0  , density(0, 2.2));

    try expectApproxEqRel(0                 , density(0, 3), eps);
    try expectApproxEqRel(0.2419707245191434, density(1, 3), eps);
    try expectApproxEqRel(0.2075537487102973, density(2, 3), eps);
}

test "chiSquared.probability" {
    try expectEqual(0, probability(-inf, 3));
    try expectEqual(1, probability( inf, 3));

    try expectApproxEqRel(0                 , probability(0, 3), eps);
    try expectApproxEqRel(0.1987480430987992, probability(1, 3), eps);
    try expectApproxEqRel(0.4275932955291208, probability(2, 3), eps);
}

test "chiSquared.quantile" {
    try expectApproxEqRel(0                , quantile(0  , 3), eps);
    try expectApproxEqRel(1.005174013052349, quantile(0.2, 3), eps);
    try expectApproxEqRel(1.869168403388716, quantile(0.4, 3), eps);
    try expectApproxEqRel(2.946166073101952, quantile(0.6, 3), eps);
    try expectApproxEqRel(4.641627676087445, quantile(0.8, 3), eps);
    try expectEqual      (inf              , quantile(1  , 3)     );
}

test "chiSquared.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.c198f554d3db5p+0, random.implementation(gen, 3), eps);
    try expectApproxEqRel(0x1.0e7afeee50b89p+1, random.implementation(gen, 3), eps);
    try expectApproxEqRel(0x1.28ce118715efcp+1, random.implementation(gen, 3), eps);
}
