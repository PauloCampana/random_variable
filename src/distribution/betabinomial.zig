//! Parameters:
//! - n: `size`   ∈ {0,1,2,⋯}
//! - α: `shape1` ∈ (0,∞)
//! - β: `shape2` ∈ (0,∞)

const std = @import("std");
// const gamma = @import("gamma.zig");
const lgamma = std.math.lgamma;
// const incompleteBeta = @import("../thirdyparty/prob.zig").incompleteBeta;
// const inverseIncompleteBeta = @import("../thirdyparty/prob.zig").inverseIncompleteBeta;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
// const isInf = std.math.isInf;
const inf = std.math.inf(f64);

pub const parameters = 3;

/// p(x) = (n x) beta(x + α, n - x + β) / beta(α, β).
pub fn density(x: f64, size: u64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(x));
    const n = @as(f64, @floatFromInt(size));
    if (x < 0 or x > n or x != @round(x)) {
        return 0;
    }
    const binom = lgamma(f64, n + 1) - lgamma(f64, x + 1) - lgamma(f64, n - x + 1);
    const beta1 = lgamma(f64, x + shape1) + lgamma(f64, n - x + shape2) - lgamma(f64, n + shape1 + shape2);
    const beta2 = lgamma(f64, shape1) + lgamma(f64, shape2) - lgamma(f64, shape1 + shape2);
    return @exp(binom + beta1 - beta2);
}

/// No closed form.
pub fn probability(q: f64, size: u64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    const n = @as(f64, @floatFromInt(size));
    if (q >= n) {
        return 1;
    }
    const p0_num = lgamma(f64, shape1 + shape2) + lgamma(f64, n + shape2);
    const p0_den = lgamma(f64, shape1 + shape2 + n) + lgamma(f64, shape2);
    var mass = @exp(p0_num - p0_den);
    var cumu: f64 = mass;
    const qu = @as(usize, @intFromFloat(q));
    for (0..qu) |x| {
        const fx = @as(f64, @floatFromInt(x));
        const num = (n - fx) * (shape1 + fx);
        const den = (fx + 1) * (shape2 + n - fx - 1);
        mass *= num / den;
        cumu += mass;
    }
    return cumu;
}

/// No closed form.
pub fn quantile(p: f64, size: u64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(0 <= p and p <= 1);
    const n = @as(f64, @floatFromInt(size));
    if (p == 0 or n == 0) {
        return 0;
    }
    if (p == 1) {
        return n;
    }
    const p0_num = lgamma(f64, shape1 + shape2) + lgamma(f64, n + shape2);
    const p0_den = lgamma(f64, shape1 + shape2 + n) + lgamma(f64, shape2);
    var mass = @exp(p0_num - p0_den);
    var cumu: f64 = mass;
    var bb: f64 = 0;
    while (p >= cumu) : (bb += 1) {
        const num = (n - bb) * (shape1 + bb);
        const den = (bb + 1) * (shape2 + n - bb - 1);
        mass *= num / den;
        cumu += mass;
    }
    return bb;
}

/// Uses the quantile function.
pub const random = struct {
    fn implementation(generator: std.rand.Random, size: u64, shape1: f64, shape2: f64) f64 {
        const n = @as(f64, @floatFromInt(size));
        if (n == 0) {
            return 0;
        }
        const p0_num = lgamma(f64, shape1 + shape2) + lgamma(f64, n + shape2);
        const p0_den = lgamma(f64, shape1 + shape2 + n) + lgamma(f64, shape2);
        var mass = @exp(p0_num - p0_den);
        var cumu: f64 = mass;
        var bb: f64 = 0;
        const uni = generator.float(f64);
        while (uni >= cumu) : (bb += 1) {
            const num = (n - bb) * (shape1 + bb);
            const den = (bb + 1) * (shape2 + n - bb - 1);
            mass *= num / den;
            cumu += mass;
        }
        return bb;
    }

    pub fn single(generator: std.rand.Random, size: u64, shape1: f64, shape2: f64) f64 {
        assert(isFinite(shape1) and isFinite(shape2));
        assert(shape1 > 0 and shape2 > 0);
        return implementation(generator, size, shape1, shape2);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, size: u64, shape1: f64, shape2: f64) []f64 {
        assert(isFinite(shape1) and isFinite(shape2));
        assert(shape1 > 0 and shape2 > 0);
        for (buf) |*x| {
            x.* =  implementation(generator, size, shape1, shape2);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, size: u64, shape1: f64, shape2: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, size, shape1, shape2);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 30 * std.math.floatEps(f64); // 6.66 × 10^-15

test "betabinomial.density" {
    try expectEqual(0, density(-inf, 10, 3, 5));
    try expectEqual(0, density( inf, 10, 3, 5));

    try expectEqual(1, density(0, 0, 3, 5));
    try expectEqual(0, density(1, 0, 3, 5));

    try expectApproxEqRel(0                  , density(-0.1, 10, 3, 5), eps);
    try expectApproxEqRel(0.05147058823529412, density( 0  , 10, 3, 5), eps);
    try expectApproxEqRel(0                  , density( 0.1, 10, 3, 5), eps);
    try expectApproxEqRel(0                  , density( 0.9, 10, 3, 5), eps);
    try expectApproxEqRel(0.11029411764705882, density( 1  , 10, 3, 5), eps);
    try expectApproxEqRel(0                  , density( 1.1, 10, 3, 5), eps);
}

test "betabinomial.probability" {
    try expectEqual(0, probability(-inf, 10, 3, 5));
    try expectEqual(1, probability( inf, 10, 3, 5));

    try expectEqual(1, probability(0, 0, 3, 5));
    try expectEqual(1, probability(1, 0, 3, 5));

    try expectApproxEqRel(0                  , probability(-0.1, 10, 3, 5), eps);
    try expectApproxEqRel(0.05147058823529412, probability( 0  , 10, 3, 5), eps);
    try expectApproxEqRel(0.05147058823529412, probability( 0.1, 10, 3, 5), eps);
    try expectApproxEqRel(0.05147058823529412, probability( 0.9, 10, 3, 5), eps);
    try expectApproxEqRel(0.16176470588235294, probability( 1  , 10, 3, 5), eps);
    try expectApproxEqRel(0.16176470588235294, probability( 1.1, 10, 3, 5), eps);
}

test "betabinomial.quantile" {
    try expectEqual(0, quantile(0  , 0, 3, 5));
    try expectEqual(0, quantile(0.5, 0, 3, 5));
    try expectEqual(0, quantile(1  , 0, 3, 5));

    try expectEqual( 0, quantile(0                , 10, 3, 5));
    try expectEqual( 0, quantile(0.051470588235292, 10, 3, 5));
    try expectEqual( 0, quantile(0.051470588235293, 10, 3, 5));
    try expectEqual( 1, quantile(0.051470588235294, 10, 3, 5));
    try expectEqual( 1, quantile(0.161764705882350, 10, 3, 5));
    try expectEqual( 1, quantile(0.161764705882351, 10, 3, 5));
    try expectEqual( 2, quantile(0.161764705882352, 10, 3, 5));
    try expectEqual(10, quantile(1                , 10, 3, 5));
}

test "betabinomial.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(0, random.implementation(gen, 0, 3, 5));
    try expectEqual(0, random.implementation(gen, 0, 3, 5));
    try expectEqual(0, random.implementation(gen, 0, 3, 5));

    try expectEqual(3, random.implementation(gen, 10, 3, 5));
    try expectEqual(3, random.implementation(gen, 10, 3, 5));
    try expectEqual(4, random.implementation(gen, 10, 3, 5));
}
