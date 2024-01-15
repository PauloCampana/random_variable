//! Support: X ∈ (-∞,∞)
//!
//! Parameters:
//! - μ: `location` ∈ (-∞,∞)
//! - σ: `scale`    ∈ ( 0,∞)

const std = @import("std");
const normalDist = @import("../thirdyparty/prob.zig").normalDist;
const inverseNormalDist = @import("../thirdyparty/prob.zig").inverseNormalDist;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = false;
pub const parameters = 2;

/// f(x) = exp(-((x - μ) / σ)^2 / 2) / (σ sqrt(2π)).
pub fn density(x: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    const z = (x - location) / scale;
    const sqrt2pi = comptime @sqrt(2 * std.math.pi);
    return @exp(-0.5 * z * z) / (scale * sqrt2pi);
}

/// No closed form.
pub fn probability(q: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(q));
    const z = (q - location) / scale;
    return normalDist(z);
}

/// No closed form.
pub fn quantile(p: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(0 <= p and p <= 1);
    const q = inverseNormalDist(p);
    return location + scale * q;
}

pub const random = struct {
    pub fn single(generator: std.rand.Random, location: f64, scale: f64) f64 {
        assert(isFinite(location) and isFinite(scale));
        assert(scale > 0);
        const nor = generator.floatNorm(f64);
        return location + scale * nor;
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, location: f64, scale: f64) []f64 {
        assert(isFinite(location) and isFinite(scale));
        assert(scale > 0);
        for (buffer) |*x| {
            const nor = generator.floatNorm(f64);
            x.* = location + scale * nor;
        }
        return buffer;
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "normal.density" {
    try expectEqual(0, density(-inf, 0, 1));
    try expectEqual(0, density( inf, 0, 1));

    try expectApproxEqRel(0.3989422804014327, density(0, 0, 1), eps);
    try expectApproxEqRel(0.2419707245191433, density(1, 0, 1), eps);
    try expectApproxEqRel(0.0539909665131880, density(2, 0, 1), eps);
}

test "normal.probability" {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0.5               , probability(0, 0, 1), eps);
    try expectApproxEqRel(0.8413447460685429, probability(1, 0, 1), eps);
    try expectApproxEqRel(0.9772498680518208, probability(2, 0, 1), eps);
}

test "normal.quantile" {
    try expectEqual      (-inf               , quantile(0  , 0, 1)     );
    try expectApproxEqRel(-0.8416212335729142, quantile(0.2, 0, 1), eps);
    try expectApproxEqRel(-0.2533471031357998, quantile(0.4, 0, 1), eps);
    try expectApproxEqRel( 0.2533471031358001, quantile(0.6, 0, 1), eps);
    try expectApproxEqRel( 0.8416212335729144, quantile(0.8, 0, 1), eps);
    try expectEqual      ( inf               , quantile(1  , 0, 1)     );
}

test "normal.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(-0x1.2bd4a0beac2dfp-2, random.single(gen, 0, 1), eps);
    try expectApproxEqRel(-0x1.6d1e253ea4858p-1, random.single(gen, 0, 1), eps);
    try expectApproxEqRel(-0x1.af653db4b3107p-4, random.single(gen, 0, 1), eps);
}
