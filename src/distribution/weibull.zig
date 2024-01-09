//! Weibull distribution.
//!
//! Parameters:
//!     α: `shape` ∈ (0,∞)
//!     λ: `rate`  ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

pub const parameters = 2;
pub const support = [2]f64 {0, inf};

/// f(x) = αλ (λx)^(α - 1) exp(-(λx)^α).
pub fn density(x: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x)) {
        return 0;
    }
    if (x == 0) {
        if (shape == 1) {
            return rate;
        }
        return if (shape < 1) inf else 0;
    }
    const z = rate * x;
    const zam1 = std.math.pow(f64, z, shape - 1);
    const za = zam1 * z;
    return shape * rate * zam1 * @exp(-za);
}

/// F(q) = 1 - exp(-(λq)^α).
pub fn probability(q: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = rate * q;
    const za = std.math.pow(f64, z, shape);
    return -std.math.expm1(-za);
}

/// Q(p) = (-ln(1 - p))^(1 / α) / λ.
pub fn quantile(p: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(0 <= p and p <= 1);
    const q1 = -std.math.log1p(-p);
    const q2 = std.math.pow(f64, q1, 1 / shape);
    return q2 / rate;
}

/// Uses the Ziggurat method and the quantile function.
pub const random = struct {
    fn implementation(generator: std.rand.Random, shape: f64, rate: f64) f64 {
        const exp = generator.floatExp(f64);
        const wei = std.math.pow(f64, exp, 1 / shape);
        return wei / rate;
    }

    pub fn single(generator: std.rand.Random, shape: f64, rate: f64) f64 {
        assert(isFinite(shape) and isFinite(rate));
        assert(shape > 0 and rate > 0);
        return implementation(generator, shape, rate);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, shape: f64, rate: f64) []f64 {
        assert(isFinite(shape) and isFinite(rate));
        assert(shape > 0 and rate > 0);
        for (buf) |*x| {
            x.* = implementation(generator, shape, rate);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, shape: f64, rate: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, shape, rate);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "weibull.density" {
    try expectEqual(0, density(-inf, 3, 0.5));
    try expectEqual(0, density( inf, 3, 0.5));

    try expectEqual(inf, density(0, 0.9, 5));
    try expectEqual(5  , density(0, 1  , 5));
    try expectEqual(0  , density(0, 1.1, 5));

    try expectApproxEqRel(0                 , density(0, 3, 0.5), eps);
    try expectApproxEqRel(0.3309363384692233, density(1, 3, 0.5), eps);
    try expectApproxEqRel(0.5518191617571635, density(2, 3, 0.5), eps);
}

test "weibull.probability" {
    try expectEqual(0, probability(-inf, 3, 0.5));
    try expectEqual(1, probability( inf, 3, 0.5));

    try expectApproxEqRel(0                 , probability(0, 3, 0.5), eps);
    try expectApproxEqRel(0.1175030974154046, probability(1, 3, 0.5), eps);
    try expectApproxEqRel(0.6321205588285577, probability(2, 3, 0.5), eps);
}

test "weibull.quantile" {
    try expectApproxEqRel(0                , quantile(0  , 3, 0.5), eps);
    try expectApproxEqRel(1.213085586248216, quantile(0.2, 3, 0.5), eps);
    try expectApproxEqRel(1.598775754926823, quantile(0.4, 3, 0.5), eps);
    try expectApproxEqRel(1.942559933595852, quantile(0.6, 3, 0.5), eps);
    try expectApproxEqRel(2.343804613759100, quantile(0.8, 3, 0.5), eps);
    try expectEqual      (inf              , quantile(1  , 3, 0.5)     );
}

test "weibull.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.29f2f11294770p+0, random.implementation(gen, 3, 0.5), eps);
    try expectApproxEqRel(0x1.479bbb94bd291p+1, random.implementation(gen, 3, 0.5), eps);
    try expectApproxEqRel(0x1.8f80c328506e1p-1, random.implementation(gen, 3, 0.5), eps);
}
