//! Parameters:
//! - n: `size` ∈ {0,1,2,⋯}
//! - p: `prob` ∈ [0,1]

const std = @import("std");
const lgamma = @import("../thirdyparty/prob.zig").lnGamma;
const incompleteBeta = @import("../thirdyparty/prob.zig").incompleteBeta;
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const parameters = 2;

/// p(x) = (n x) p^x (1 - p)^(n - x).
pub fn density(x: f64, size: u64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(!isNan(x));
    const n = @as(f64, @floatFromInt(size));
    if (x < 0 or x > n or x != @round(x)) {
        return 0;
    }
    if (prob == 0) {
        return if (x == 0) 1 else 0;
    }
    if (prob == 1) {
        return if (x == n) 1 else 0;
    }
    const diff = n - x;
    const binom = lgamma(n + 1) - lgamma(x + 1) - lgamma(diff + 1);
    const log = binom + x * @log(prob) + diff * std.math.log1p(-prob);
    return @exp(log);
}

/// No closed form.
pub fn probability(q: f64, size: u64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(!isNan(q));
    const n = @as(f64, @floatFromInt(size));
    if (q < 0) {
        return 0;
    }
    if (q >= n) {
        return 1;
    }
    if (prob == 0) {
        return 1;
    }
    if (prob == 1) {
        return 0;
    }
    const fq = @floor(q);
    return incompleteBeta(n - fq, fq + 1, 1 - prob);
}

/// No closed form
pub fn quantile(p: f64, size: u64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return 0;
    }
    const n = @as(f64, @floatFromInt(size));
    if (p == 1 or prob == 1) {
        return n;
    }
    const pq = prob / (1 - prob);
    var mass = std.math.pow(f64, 1 - prob, n);
    var cumu = mass;
    var bin: f64 = 0;
    while (p >= cumu) : (bin += 1) {
        mass *= pq * (n - bin) / (bin + 1);
        cumu += mass;
    }
    return bin;
}

/// Uses the quantile function.
pub const random = struct {
    fn implementation(generator: std.rand.Random, size: u64, prob: f64) f64 {
        const n = @as(f64, @floatFromInt(size));
        if (prob == 1) {
            return n;
        }
        const pq = prob / (1 - prob);
        var mass = std.math.pow(f64, 1 - prob, n);
        var cumu = mass;
        var bin: f64 = 0;
        const uni = generator.float(f64);
        while (uni >= cumu) : (bin += 1) {
            mass *= pq * (n - bin) / (bin + 1);
            cumu += mass;
        }
        return bin;
    }

    pub fn single(generator: std.rand.Random, size: u64, prob: f64) f64 {
        assert(0 <= prob and prob <= 1);
        return implementation(generator, size, prob);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, size: u64, prob: f64) []f64 {
        assert(0 <= prob and prob <= 1);
        for (buf) |*x| {
            x.* =  implementation(generator, size, prob);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, size: u64, prob: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, size, prob);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "binomial.density" {
    try expectEqual(0, density(-inf, 10, 0.2));
    try expectEqual(0, density( inf, 10, 0.2));

    try expectEqual(1, density( 0,  0, 0.2));
    try expectEqual(0, density( 1,  0, 0.2));
    try expectEqual(1, density( 0, 10, 0  ));
    try expectEqual(0, density( 1, 10, 0  ));
    try expectEqual(0, density( 9, 10, 1  ));
    try expectEqual(1, density(10, 10, 1  ));
    try expectEqual(0, density(11, 10, 1  ));

    try expectApproxEqRel(0           , density(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.1073741824, density( 0  , 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(0.2684354560, density( 1  , 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 1.1, 10, 0.2), eps);
}

test "binomial.probability" {
    try expectEqual(0, probability(-inf, 10, 0.2));
    try expectEqual(1, probability( inf, 10, 0.2));

    try expectEqual(1, probability( 0,  0, 0.2));
    try expectEqual(1, probability( 1,  0, 0.2));
    try expectEqual(1, probability( 0, 10, 0  ));
    try expectEqual(1, probability( 1, 10, 0  ));
    try expectEqual(0, probability( 9, 10, 1  ));
    try expectEqual(1, probability(10, 10, 1  ));
    try expectEqual(1, probability(11, 10, 1  ));

    try expectApproxEqRel(0           , probability(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.1073741824, probability( 0  , 10, 0.2), eps);
    try expectApproxEqRel(0.1073741824, probability( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.1073741824, probability( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(0.3758096384, probability( 1  , 10, 0.2), eps);
    try expectApproxEqRel(0.3758096384, probability( 1.1, 10, 0.2), eps);
}

test "binomial.quantile" {
    try expectEqual( 0, quantile(0  ,  0, 0.2));
    try expectEqual( 0, quantile(0.5,  0, 0.2));
    try expectEqual( 0, quantile(1  ,  0, 0.2));
    try expectEqual( 0, quantile(0  , 10, 0  ));
    try expectEqual( 0, quantile(0.5, 10, 0  ));
    try expectEqual(10, quantile(1  , 10, 0  ));
    try expectEqual( 0, quantile(0  , 10, 1  ));
    try expectEqual(10, quantile(0.5, 10, 1  ));
    try expectEqual(10, quantile(1  , 10, 1  ));

    try expectEqual( 0, quantile(0           , 10, 0.2));
    try expectEqual( 0, quantile(0.1073741823, 10, 0.2));
    try expectEqual( 0, quantile(0.1073741824, 10, 0.2));
    try expectEqual( 1, quantile(0.1073741825, 10, 0.2));
    try expectEqual( 1, quantile(0.3758096383, 10, 0.2));
    try expectEqual( 1, quantile(0.3758096384, 10, 0.2));
    try expectEqual( 2, quantile(0.3758096385, 10, 0.2));
    try expectEqual(10, quantile(1           , 10, 0.2));
}

test "binomial.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual( 0, random.implementation(gen,  0, 0.2));
    try expectEqual( 0, random.implementation(gen,  0, 0.2));
    try expectEqual( 0, random.implementation(gen,  0, 0.2));
    try expectEqual( 0, random.implementation(gen, 10, 0  ));
    try expectEqual( 0, random.implementation(gen, 10, 0  ));
    try expectEqual( 0, random.implementation(gen, 10, 0  ));
    try expectEqual(10, random.implementation(gen, 10, 1  ));
    try expectEqual(10, random.implementation(gen, 10, 1  ));
    try expectEqual(10, random.implementation(gen, 10, 1  ));

    try expectEqual(2, random.implementation(gen, 10, 0.2));
    try expectEqual(2, random.implementation(gen, 10, 0.2));
    try expectEqual(2, random.implementation(gen, 10, 0.2));
}
