//! Parameters:
//! - b: `base` ∈ {2,3,4,⋯}

const std = @import("std");
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const parameters = 1;

/// p(x) = log_b(1 + 1 / x).
pub fn density(x: f64, base: u64) f64 {
    assert(base >= 2);
    assert(!isNan(x));
    const fbase = @as(f64, @floatFromInt(base));
    if (x < 1 or x > fbase - 1 or x != @round(x)) {
        return 0;
    }
    return std.math.log(f64, fbase, 1 + 1 / x);
}

/// F(q) = log_b(1 + ⌊q⌋).
pub fn probability(q: f64, base: u64) f64 {
    assert(base >= 2);
    assert(!isNan(q));
    const fbase = @as(f64, @floatFromInt(base));
    if (q < 1) {
        return 0;
    }
    if (q >= fbase - 1) {
        return 1;
    }
    return std.math.log(f64, fbase, 1 + @floor(q));
}

/// Q(p) = ⌈b^p⌉ - 1.
pub fn quantile(p: f64, base: u64) f64 {
    assert(base >= 2);
    assert(0 <= p and p <= 1);
    const fbase = @as(f64, @floatFromInt(base));
    if (p == 0) {
        return 1;
    }
    const bp = std.math.pow(f64, fbase, p);
    return @ceil(bp) - 1;
}

/// Uses the quantile function.
pub const random = struct {
    fn implementation(generator: std.rand.Random, base: u64) f64 {
        const uni = generator.float(f64);
        const bp = std.math.pow(f64, @floatFromInt(base), uni);
        return @ceil(bp) - 1;
    }

    pub fn single(generator: std.rand.Random, base: u64) f64 {
        assert(base >= 2);
        return implementation(generator, base);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, base: u64) []f64 {
        assert(base >= 2);
        for (buf) |*x| {
            x.* =  implementation(generator, base);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, base: u64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, base);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "benford.density" {
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

test "benford.probability" {
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

test "benford.quantile" {
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

test "benford.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(1, random.implementation(gen,  2));
    try expectEqual(1, random.implementation(gen,  2));
    try expectEqual(1, random.implementation(gen,  2));

    try expectEqual(1, random.implementation(gen, 10));
    try expectEqual(2, random.implementation(gen, 10));
    try expectEqual(1, random.implementation(gen, 10));
}
