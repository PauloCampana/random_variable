//! Parameters:
//! - λ: `scale` ∈ (0,∞)

const std = @import("std");
const lgamma = @import("../thirdyparty/prob.zig").lnGamma;
const incompleteGamma = @import("../thirdyparty/prob.zig").incompleteGamma;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

pub const parameters = 1;
pub const support = [2]f64 {0, inf};

/// p(x) = λ^x exp(-λ) / x!.
pub fn density(x: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x) or x != @round(x)) {
        return 0;
    }
    const log = -lambda + x * @log(lambda) - lgamma(x + 1);
    return @exp(log);
}

/// No closed form.
pub fn probability(q: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    if (isInf(q)) {
        return 1;
    }
    return 1 - incompleteGamma(@floor(q) + 1, lambda);
}

/// No closed form.
pub fn quantile(p: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(0 <= p and p <= 1);
    if (p == 1) {
        return inf;
    }
    var mass = @exp(-lambda);
    var cumu = mass;
    var poi: f64 = 1;
    while (p >= cumu) : (poi += 1) {
        mass *= lambda / poi;
        cumu += mass;
    }
    return poi - 1;
}

/// Uses the quantile function.
pub const random = struct {
    fn implementation(generator: std.rand.Random, lambda: f64) f64 {
        const uni = generator.float(f64);
        var mass = @exp(-lambda);
        var cumu = mass;
        var poi: f64 = 1;
        while (uni >= cumu) : (poi += 1) {
            mass *= lambda / poi;
            cumu += mass;
        }
        return poi - 1;
    }

    pub fn single(generator: std.rand.Random, lambda: f64) f64 {
        assert(isFinite(lambda));
        assert(lambda > 0);
        return implementation(generator, lambda);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, lambda: f64) []f64 {
        assert(isFinite(lambda));
        assert(lambda > 0);
        for (buf) |*x| {
            x.* = implementation(generator, lambda);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, lambda: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, lambda);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "poisson.density" {
    try expectEqual(0, density(-inf, 3));
    try expectEqual(0, density( inf, 3));

    try expectApproxEqRel(0                 , density(-0.1, 3), eps);
    try expectApproxEqRel(0.0497870683678639, density( 0  , 3), eps);
    try expectApproxEqRel(0                 , density( 0.1, 3), eps);
    try expectApproxEqRel(0                 , density( 0.9, 3), eps);
    try expectApproxEqRel(0.1493612051035919, density( 1  , 3), eps);
    try expectApproxEqRel(0                 , density( 1.1, 3), eps);
}

test "poisson.probability" {
    try expectEqual(0, probability(-inf, 3));
    try expectEqual(1, probability( inf, 3));

    try expectApproxEqRel(0                 , probability(-0.1, 3), eps);
    try expectApproxEqRel(0.0497870683678639, probability( 0  , 3), eps);
    try expectApproxEqRel(0.0497870683678639, probability( 0.1, 3), eps);
    try expectApproxEqRel(0.0497870683678639, probability( 0.9, 3), eps);
    try expectApproxEqRel(0.1991482734714558, probability( 1  , 3), eps);
    try expectApproxEqRel(0.1991482734714558, probability( 1.1, 3), eps);
}

test "poisson.quantile" {
    try expectEqual(  0, quantile(0                 , 3));
    try expectEqual(  0, quantile(0.0497870683678638, 3));
    try expectEqual(  0, quantile(0.0497870683678639, 3));
    try expectEqual(  1, quantile(0.0497870683678640, 3));
    try expectEqual(  1, quantile(0.1991482734714556, 3));
    try expectEqual(  1, quantile(0.1991482734714557, 3));
    try expectEqual(  2, quantile(0.1991482734714558, 3));
    try expectEqual(inf, quantile(1                 , 3));
}

test "poisson.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(2, random.implementation(gen, 3));
    try expectEqual(2, random.implementation(gen, 3));
    try expectEqual(3, random.implementation(gen, 3));
}
