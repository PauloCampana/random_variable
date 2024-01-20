//! Support: X ∈ {1,2,3,⋯}
//!
//! Parameters:
//! - p: `prob` ∈ (0,1)

const std = @import("std");
const incompleteBeta = @import("../thirdyparty/prob.zig").incompleteBeta;
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = true;
pub const parameters = 1;

/// p(x) = p^x / (-ln(1 - p) x).
pub fn density(x: f64, prob: f64) f64 {
    assert(0 < prob and prob < 1);
    assert(!isNan(x));
    if (x < 1 or x != @round(x)) {
        return 0;
    }
    const pow = std.math.pow(f64, prob, x);
    const ln = -std.math.log1p(-prob);
    return pow / (x * ln);
}

/// No closed form.
pub fn probability(q: f64, prob: f64) f64 {
    assert(0 < prob and prob < 1);
    assert(!isNan(q));
    if (q < 1) {
        return 0;
    }
    if (q == inf) {
        return 1;
    }
    var mass = prob / -std.math.log1p(-prob);
    var cumu = mass;
    var loga: f64 = 1;
    for (1..@intFromFloat(q)) |_| {
        const num = prob * loga;
        loga += 1;
        mass *= num / loga;
        cumu += mass;
    }
    return cumu;
}

/// No closed form
pub fn quantile(p: f64, prob: f64) f64 {
    assert(0 < prob and prob < 1);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return 1;
    }
    if (p == 1) {
        return inf;
    }
    const initial_mass = prob / -std.math.log1p(-prob);
    return linearSearch(p, prob, initial_mass);
}

/// Uses the quantile function.
pub const random = struct {
    pub fn single(generator: std.rand.Random, prob: f64) f64 {
        assert(0 < prob and prob < 1);
        const r = std.math.log1p(-prob);
        return kemp(generator, prob, r);
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, prob: f64) []f64 {
        assert(0 < prob and prob < 1);
        const r = std.math.log1p(-prob);
        for (buffer) |*x| {
            x.* = kemp(generator, prob, r);
        }
        return buffer;
    }
};

inline fn linearSearch(p: f64, prob: f64, initial_mass: f64) f64 {
    var mass = initial_mass;
    var cumu = mass;
    var loga: f64 = 1;
    while (cumu <= p) {
        const num = prob * loga;
        loga += 1;
        mass *= num / loga;
        cumu += mass;
    }
    return loga;
}

inline fn kemp(generator: std.rand.Random, prob: f64, r: f64) f64 {
    const uni1 = generator.float(f64);
    if (uni1 > prob) {
        return 1;
    }
    const uni2 = generator.float(f64);
    const q = -std.math.expm1(r * uni2);
    if (uni1 > q) {
        return 2;
    }
    if (q * q < uni1 and uni1 <= q) {
        return 1;
    }
    return @floor(1 + @log(uni1) / @log(q));
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "logarithmic.density" {
    try expectEqual(0, density(-inf, 0.2));
    try expectEqual(0, density( inf, 0.2));

    try expectApproxEqRel(0                 , density(0.9, 0.2), eps);
    try expectApproxEqRel(0.8962840235449099, density(1  , 0.2), eps);
    try expectApproxEqRel(0                 , density(1.1, 0.2), eps);
    try expectApproxEqRel(0                 , density(1.9, 0.2), eps);
    try expectApproxEqRel(0.0896284023544909, density(2  , 0.2), eps);
    try expectApproxEqRel(0                 , density(2.1, 0.2), eps);
}

test "logarithmic.probability" {
    try expectEqual(0, probability(-inf, 0.2));
    try expectEqual(1, probability( inf, 0.2));

    try expectApproxEqRel(0                 , probability(0.9, 0.2), eps);
    try expectApproxEqRel(0.8962840235449099, probability(1  , 0.2), eps);
    try expectApproxEqRel(0.8962840235449099, probability(1.1, 0.2), eps);
    try expectApproxEqRel(0.8962840235449099, probability(1.9, 0.2), eps);
    try expectApproxEqRel(0.9859124258994009, probability(2  , 0.2), eps);
    try expectApproxEqRel(0.9859124258994009, probability(2.1, 0.2), eps);
}

test "logarithmic.quantile" {
    try expectEqual(  1, quantile(0                 , 0.2));
    try expectEqual(  1, quantile(0.8962840235449098, 0.2));
    try expectEqual(  1, quantile(0.8962840235449099, 0.2));
    try expectEqual(  2, quantile(0.8962840235449100, 0.2));
    try expectEqual(  2, quantile(0.9859124258994008, 0.2));
    try expectEqual(  2, quantile(0.9859124258994009, 0.2));
    try expectEqual(  3, quantile(0.9859124258994010, 0.2));
    try expectEqual(inf, quantile(1                 , 0.2));
}

test "logarithmic.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(1, random.single(gen, 0.2));
    try expectEqual(1, random.single(gen, 0.2));
    try expectEqual(1, random.single(gen, 0.2));
}
