//! Support: X ∈ {0,1,2,⋯}
//!
//! Parameters:
//! - n: `size` ∈ {0,1,2,⋯}
//! - p: `prob` ∈ (0,1]

const std = @import("std");
const math = @import("../math.zig");
const incompleteBeta = @import("../thirdyparty/prob.zig").incompleteBeta;
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = true;
pub const parameters = 2;

/// p(x) = (x + n - 1 x) p^n (1 - p)^x.
pub fn density(x: f64, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    assert(!isNan(x));
    if (x < 0 or x == inf or x != @round(x)) {
        return 0;
    }
    if (prob == 1) {
        return if (x == 0) 1 else 0;
    }
    const n = @as(f64, @floatFromInt(size));
    const binom = math.lbinomial(n + x - 1, x);
    const log = binom + n * @log(prob) + x * std.math.log1p(-prob);
    return @exp(log);
}

/// No closed form.
pub fn probability(q: f64, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    if (q == inf or prob == 1) {
        return 1;
    }
    const n = @as(f64, @floatFromInt(size));
    return incompleteBeta(n, @floor(q) + 1, prob);
}

/// No closed form.
pub fn quantile(p: f64, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    assert(0 <= p and p <= 1);
    if (p == 0 or prob == 1) {
        return 0;
    }
    if (p == 1) {
        return inf;
    }
    const n = @as(f64, @floatFromInt(size));
    const q = 1 - prob;
    var mass = std.math.pow(f64, prob, n);
    var cumu = mass;
    var nbi: f64 = 0;
    while (p >= cumu) : (nbi += 1) {
        mass *= q * (n + nbi) / (nbi + 1);
        cumu += mass;
    }
    return nbi;
}

/// Uses the quantile function.
pub const random = struct {
    pub fn single(generator: std.rand.Random, size: u64, prob: f64) f64 {
        assert(0 < prob and prob <= 1);
        assert(size != 0);
        if (prob == 1) {
            return 0;
        }
        const n = @as(f64, @floatFromInt(size));
        const q = 1 - prob;
        const initial_mass = std.math.pow(f64, prob, n);
        const uni = generator.float(f64);
        return linearSearch(uni, n, q, initial_mass);
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, size: u64, prob: f64) []f64 {
        assert(0 < prob and prob <= 1);
        assert(size != 0);
        if (prob == 1) {
            @memset(buffer, 0);
            return buffer;
        }
        const n = @as(f64, @floatFromInt(size));
        const q = 1 - prob;
        const initial_mass = std.math.pow(f64, prob, n);
        for (buffer) |*x| {
            const uni = generator.float(f64);
            x.* = linearSearch(uni, n, q, initial_mass);
        }
        return buffer;
    }
};

inline fn linearSearch(p: f64, n: f64, q: f64, initial_mass: f64) f64 {
    var nbin: f64 = 0;
    var mass = initial_mass;
    var cumu = mass;
    while (cumu <= p) {
        const num = q * (n + nbin);
        nbin += 1;
        mass *= num / nbin;
        cumu += mass;
    }
    return nbin;
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "negativeBinomial.density" {
    try expectEqual(0, density(-inf, 10, 0.2));
    try expectEqual(0, density( inf, 10, 0.2));

    try expectEqual(1, density(0, 10, 1));
    try expectEqual(0, density(1, 10, 1));

    try expectApproxEqRel(0           , density(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.0000001024, density( 0  , 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(0.0000008192, density( 1  , 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 1.1, 10, 0.2), eps);
}

test "negativeBinomial.probability" {
    try expectEqual(0, probability(-inf, 10, 0.2));
    try expectEqual(1, probability( inf, 10, 0.2));

    try expectEqual(1, probability(0, 10, 1));
    try expectEqual(1, probability(1, 10, 1));

    try expectApproxEqRel(0           , probability(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.0000001024, probability( 0  , 10, 0.2), eps);
    try expectApproxEqRel(0.0000001024, probability( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.0000001024, probability( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(0.0000009216, probability( 1  , 10, 0.2), eps);
    try expectApproxEqRel(0.0000009216, probability( 1.1, 10, 0.2), eps);
}

test "negativeBinomial.quantile" {
    try expectEqual(0, quantile(0  , 10, 1));
    try expectEqual(0, quantile(0.5, 10, 1));
    try expectEqual(0, quantile(1  , 10, 1));

    try expectEqual(  0, quantile(0           , 10, 0.2));
    try expectEqual(  0, quantile(0.0000001023, 10, 0.2));
    try expectEqual(  0, quantile(0.0000001024, 10, 0.2));
    try expectEqual(  1, quantile(0.0000001025, 10, 0.2));
    try expectEqual(  1, quantile(0.0000009215, 10, 0.2));
    try expectEqual(  1, quantile(0.0000009216, 10, 0.2));
    try expectEqual(  2, quantile(0.0000009217, 10, 0.2));
    try expectEqual(inf, quantile(1           , 10, 0.2));
}

test "negativeBinomial.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(0, random.single(gen, 10, 1));
    try expectEqual(0, random.single(gen, 10, 1));
    try expectEqual(0, random.single(gen, 10, 1));

    try expectEqual(34, random.single(gen, 10, 0.2));
    try expectEqual(36, random.single(gen, 10, 0.2));
    try expectEqual(38, random.single(gen, 10, 0.2));
}
