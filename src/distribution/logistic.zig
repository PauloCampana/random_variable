//! Logistic distribution.
//!
//! Parameters:
//!     μ: `location` ∈ (-∞,∞)
//!     σ: `scale`    ∈ ( 0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const parameters = 2;
pub const support = [2]f64 {-inf, inf};

/// f(x) = exp(-(x - μ) / σ) / (σ (1 + exp(-(x - μ) / σ))^2).
pub fn density(x: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    const z = @abs(x - location) / scale;
    const expz = @exp(-z);
    const expzp1 = expz + 1;
    return expz / (scale * expzp1 * expzp1);
}

/// F(q) = 1 / (1 + exp(-(x - μ) / σ)).
pub fn probability(q: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(q));
    const z = (q - location) / scale;
    return 1 / (1 + @exp(-z));
}

/// Q(p) = μ + σ ln(p / (1 - p))
pub fn quantile(p: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(0 <= p and p <= 1);
    const q = @log(p / (1 - p));
    return location + scale * q;
}

/// Uses the quantile function.
const random = struct {
    fn implementation(generator: std.rand.Random, location: f64, scale: f64) f64 {
        const uni = generator.float(f64);
        return location + scale * @log(uni / (1 - uni));
    }

    pub fn single(generator: std.rand.Random, location: f64, scale: f64) f64 {
        assert(isFinite(location) and isFinite(scale));
        assert(scale > 0);
        return implementation.logistic(generator, location, scale);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, location: f64, scale: f64) []f64 {
        assert(isFinite(location) and isFinite(scale));
        assert(scale > 0);
        for (buf) |*x| {
            x.* = implementation.logistic(generator, location, scale);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, location: f64, scale: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, location, scale);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "logistic.density" {
    try expectEqual(@as(f64, 0), density(-inf, 0, 1));
    try expectEqual(@as(f64, 0), density( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0.25              ), density(0, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.1966119332414819), density(1, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.1049935854035065), density(2, 0, 1), eps);
}

test "logistic.probability" {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0.5               , probability(0, 0, 1), eps);
    try expectApproxEqRel(0.7310585786300049, probability(1, 0, 1), eps);
    try expectApproxEqRel(0.8807970779778823, probability(2, 0, 1), eps);
}

test "logistic.quantile" {
    try expectEqual      (-inf               , quantile(0  , 0, 1)     );
    try expectApproxEqRel(-1.3862943611198906, quantile(0.2, 0, 1), eps);
    try expectApproxEqRel(-0.4054651081081643, quantile(0.4, 0, 1), eps);
    try expectApproxEqRel( 0.4054651081081648, quantile(0.6, 0, 1), eps);
    try expectApproxEqRel( 1.3862943611198908, quantile(0.8, 0, 1), eps);
    try expectEqual      ( inf               , quantile(1  , 0, 1)     );
}

test "logistic.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, -0x1.1b5814cb6fc9ap-1), random.implementation(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, -0x1.67d902cb3c67ep-2), random.implementation(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, -0x1.0370f3fe2a1a1p-7), random.implementation(gen, 0, 1), eps);
}
