//! Support: X ∈ (-∞,∞)
//!
//! Parameters:
//! - μ: `location` ∈ (-∞,∞)
//! - σ: `scale`    ∈ ( 0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = false;
pub const parameters = 2;

/// f(x) = 1 / (πσ (1 + ((x - μ) / σ)^2)).
pub fn density(x: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    const z = (x - location) / scale;
    return 1 / (std.math.pi * scale * (1 + z * z));
}

/// F(q) = 0.5 + arctan((q - μ) / σ) / π.
pub fn probability(q: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(q));
    const z = (q - location) / scale;
    return 0.5 + std.math.atan(z) / std.math.pi;
}

/// Q(p) = μ + σ tan(π (p - 0.5)).
pub fn quantile(p: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return -inf;
    }
    if (p == 1) {
        return inf;
    }
    const q = @tan(std.math.pi * (p - 0.5));
    return location + scale * q;
}

/// Uses the quantile function.
pub const random = struct {
    fn implementation(generator: std.rand.Random, location: f64, scale: f64) f64 {
        const uni = generator.float(f64);
        return location + scale * @tan(std.math.pi * uni);
    }

    pub fn single(generator: std.rand.Random, location: f64, scale: f64) f64 {
        assert(isFinite(location) and isFinite(scale));
        assert(scale > 0);
        return implementation(generator, location, scale);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, location: f64, scale: f64) []f64 {
        assert(isFinite(location) and isFinite(scale));
        assert(scale > 0);
        for (buf) |*x| {
            x.* = implementation(generator, location, scale);
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

test "cauchy.density" {
    try expectEqual(0, density(-inf, 0, 1));
    try expectEqual(0, density( inf, 0, 1));

    try expectApproxEqRel(0.3183098861837906, density(0, 0, 1), eps);
    try expectApproxEqRel(0.1591549430918953, density(1, 0, 1), eps);
    try expectApproxEqRel(0.0636619772367581, density(2, 0, 1), eps);
}

test "cauchy.probability" {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0.5               , probability(0, 0, 1), eps);
    try expectApproxEqRel(0.75              , probability(1, 0, 1), eps);
    try expectApproxEqRel(0.8524163823495667, probability(2, 0, 1), eps);
}

test "cauchy.quantile" {
    try expectEqual      (-inf               , quantile(0  , 0, 1)     );
    try expectApproxEqRel(-1.3763819204711736, quantile(0.2, 0, 1), eps);
    try expectApproxEqRel(-0.3249196962329063, quantile(0.4, 0, 1), eps);
    try expectApproxEqRel( 0.3249196962329066, quantile(0.6, 0, 1), eps);
    try expectApproxEqRel( 1.3763819204711740, quantile(0.8, 0, 1), eps);
    try expectEqual      ( inf               , quantile(1  , 0, 1)     );
}

test "cauchy.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.1baa5d88fd11ap+1, random.implementation(gen, 0, 1), eps);
    try expectApproxEqRel(0x1.c8d1141faf950p+1, random.implementation(gen, 0, 1), eps);
    try expectApproxEqRel(0x1.419f9beb83432p+7, random.implementation(gen, 0, 1), eps);
}
