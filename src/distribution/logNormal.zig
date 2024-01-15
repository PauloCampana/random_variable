//! Support: X ∈ [0,∞)
//!
//! Parameters:
//! - μ: `log_location` ∈ (-∞,∞)
//! - σ: `log_scale`    ∈ ( 0,∞)

const std = @import("std");
const normalDist = @import("../thirdyparty/prob.zig").normalDist;
const inverseNormalDist = @import("../thirdyparty/prob.zig").inverseNormalDist;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = false;
pub const parameters = 2;

/// f(x) = exp(-((ln(x) - μ) / σ)^2 / 2) / (xσ sqrt(2π)).
pub fn density(x: f64, log_location: f64, log_scale: f64) f64 {
    assert(isFinite(log_location) and isFinite(log_scale));
    assert(log_scale > 0);
    assert(!isNan(x));
    if (x <= 0) {
        return 0;
    }
    const z = (@log(x) - log_location) / log_scale;
    const sqrt2pi = comptime @sqrt(2 * std.math.pi);
    return @exp(-0.5 * z * z) / (x * log_scale * sqrt2pi);
}

/// No closed form.
pub fn probability(q: f64, log_location: f64, log_scale: f64) f64 {
    assert(isFinite(log_location) and isFinite(log_scale));
    assert(log_scale > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = (@log(q) - log_location) / log_scale;
    return normalDist(z);
}

/// No closed form
pub fn quantile(p: f64, log_location: f64, log_scale: f64) f64 {
    assert(isFinite(log_location) and isFinite(log_scale));
    assert(log_scale > 0);
    assert(0 <= p and p <= 1);
    const q = inverseNormalDist(p);
    return @exp(log_location + log_scale * q);
}

/// Uses the relation to Normal distribution.
pub const random = struct {
    pub fn single(generator: std.rand.Random, log_location: f64, log_scale: f64) f64 {
        assert(isFinite(log_location) and isFinite(log_scale));
        assert(log_scale > 0);
        const nor = generator.floatNorm(f64);
        const log = log_location + log_scale * nor;
        return @exp(log);
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, log_location: f64, log_scale: f64) []f64 {
        assert(isFinite(log_location) and isFinite(log_scale));
        assert(log_scale > 0);
        for (buffer) |*x| {
            const nor = generator.floatNorm(f64);
            const log = log_location + log_scale * nor;
            x.* = @exp(log);
        }
        return buffer;
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "logNormal.density" {
    try expectEqual(0, density(-inf, 0, 1));
    try expectEqual(0, density( inf, 0, 1));

    try expectApproxEqRel(0                 , density(0, 0, 1), eps);
    try expectApproxEqRel(0.3989422804014327, density(1, 0, 1), eps);
    try expectApproxEqRel(0.1568740192789811, density(2, 0, 1), eps);
}

test "logNormal.probability" {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0                 , probability(0, 0, 1), eps);
    try expectApproxEqRel(0.5               , probability(1, 0, 1), eps);
    try expectApproxEqRel(0.7558914042144173, probability(2, 0, 1), eps);
}

test "quantile.logNormal" {
    try expectApproxEqRel(0                 , quantile(0  , 0, 1), eps);
    try expectApproxEqRel(0.4310111868818386, quantile(0.2, 0, 1), eps);
    try expectApproxEqRel(0.7761984141563506, quantile(0.4, 0, 1), eps);
    try expectApproxEqRel(1.2883303827500079, quantile(0.6, 0, 1), eps);
    try expectApproxEqRel(2.3201253945043181, quantile(0.8, 0, 1), eps);
    try expectEqual      (inf               , quantile(1  , 0, 1)     );
}

test "logNormal.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.7e09d992a530ep-1, random.single(gen, 0, 1), eps);
    try expectApproxEqRel(0x1.f5e0036c64e29p-2, random.single(gen, 0, 1), eps);
    try expectApproxEqRel(0x1.ccd17150549b1p-1, random.single(gen, 0, 1), eps);
}
