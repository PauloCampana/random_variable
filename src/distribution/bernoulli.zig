//! Parameters:
//! - p: `prob` ∈ [0,1]

const std = @import("std");
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const parameters = 1;
pub const support = [2]f64 {0, 1};

/// p(x) = 1 - p, x = 0,
///
/// p(x) = p    , x = 1.
pub fn density(x: f64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(!isNan(x));
    if (x == 0) {
        return 1 - prob;
    }
    if (x == 1) {
        return prob;
    }
    return 0;
}

/// F(q) = 0    ,      q < 0,
///
/// F(q) = 1 - p, 0 <= q < 1,
///
/// F(q) = 1    , 1 <= q.
pub fn probability(q: f64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    if (q < 1) {
        return 1 - prob;
    }
    return 1;
}

/// Q(x) = 0, x <= 1 - p,
/// Q(x) = 1, x >  1 - p.
pub fn quantile(p: f64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(0 <= p and p <= 1);
    return if (p > 1 - prob) 1 else 0;
}

/// Uses the quantile function.
pub const random = struct {
    fn implementation(generator: std.rand.Random, prob: f64) f64 {
        const uni = generator.float(f64);
        const ber = @intFromBool(uni < prob);
        return @floatFromInt(ber);
    }

    pub fn single(generator: std.rand.Random, prob: f64) f64 {
        assert(0 <= prob and prob <= 1);
        return implementation(generator, prob);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, prob: f64) []f64 {
        assert(0 <= prob and prob <= 1);
        for (buf) |*x| {
            x.* = implementation(generator, prob);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, prob: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, prob);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "bernoulli.density" {
    try expectEqual(0, density(-inf, 0.2));
    try expectEqual(0, density( inf, 0.2));

    try expectApproxEqRel(0  , density(-0.1, 0.2), eps);
    try expectApproxEqRel(0.8, density( 0  , 0.2), eps);
    try expectApproxEqRel(0  , density( 0.1, 0.2), eps);
    try expectApproxEqRel(0  , density( 0.9, 0.2), eps);
    try expectApproxEqRel(0.2, density( 1  , 0.2), eps);
    try expectApproxEqRel(0  , density( 1.1, 0.2), eps);
}

test "bernoulli.probability" {
    try expectEqual(0, probability(-inf, 0.2));
    try expectEqual(1, probability( inf, 0.2));

    try expectApproxEqRel(0  , probability(-0.1, 0.2), eps);
    try expectApproxEqRel(0.8, probability( 0  , 0.2), eps);
    try expectApproxEqRel(0.8, probability( 0.1, 0.2), eps);
    try expectApproxEqRel(0.8, probability( 0.9, 0.2), eps);
    try expectApproxEqRel(1  , probability( 1  , 0.2), eps);
    try expectApproxEqRel(1  , probability( 1.1, 0.2), eps);
}

test "bernoulli.quantile" {
    try expectEqual(0, quantile(0   , 0.2));
    try expectEqual(0, quantile(0.79, 0.2));
    try expectEqual(0, quantile(0.8 , 0.2));
    try expectEqual(1, quantile(0.81, 0.2));
    try expectEqual(1, quantile(1   , 0.2));
}

test "bernoulli.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(0, random.implementation(gen, 0));
    try expectEqual(0, random.implementation(gen, 0));
    try expectEqual(0, random.implementation(gen, 0));
    try expectEqual(1, random.implementation(gen, 1));
    try expectEqual(1, random.implementation(gen, 1));
    try expectEqual(1, random.implementation(gen, 1));

    try expectEqual(0, random.implementation(gen, 0.2));
    try expectEqual(0, random.implementation(gen, 0.2));
    try expectEqual(0, random.implementation(gen, 0.2));
}
