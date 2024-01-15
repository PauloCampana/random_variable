//! Support: X ∈ [0,∞)
//!
//! Parameters:
//! - α: `shape1` ∈ (0,∞)
//! - β: `shape2` ∈ (0,∞)

const std = @import("std");
const gamma = @import("gamma.zig");
const math = @import("../math.zig");
const incompleteBeta = @import("../thirdyparty/prob.zig").incompleteBeta;
const inverseIncompleteBeta = @import("../thirdyparty/prob.zig").inverseIncompleteBeta;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

pub const discrete = false;
pub const parameters = 2;

/// f(x) = x^(α - 1) (1 + x)^(-α - β) / beta(α, β).
pub fn density(x: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x)) {
        return 0;
    }
    if (x == 0) {
        if (shape1 == 1) {
            return shape2;
        }
        return if (shape1 < 1) inf else 0;
    }
    const num = (shape1 - 1) * @log(x) - (shape1 + shape2) * std.math.log1p(x);
    const den = math.lbeta(shape1, shape2);
    return @exp(num - den);
}

/// No closed form.
pub fn probability(q: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    if (isInf(q)) {
        return 1;
    }
    const z = q / (1 + q);
    return incompleteBeta(shape1, shape2, z);
}

/// No closed form.
pub fn quantile(p: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(0 <= p and p <= 1);
    const q = inverseIncompleteBeta(shape1, shape2, p);
    return q / (1 - q);
}

/// Uses the relation to Gamma distribution.
pub const random = struct {
    pub fn single(generator: std.rand.Random, shape1: f64, shape2: f64) f64 {
        const gam1 = gamma.random.single(generator, shape1, 1);
        const gam2 = gamma.random.single(generator, shape2, 1);
        return gam1 / gam2;
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, shape1: f64, shape2: f64) []f64 {
        for (buffer) |*x| {
            const gam1 = gamma.random.single(generator, shape1, 1);
            const gam2 = gamma.random.single(generator, shape2, 1);
            x.* = gam1 / gam2;
        }
        return buffer;
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "betaPrime.density" {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectEqual(inf, density(0, 0.9, 5));
    try expectEqual(  5, density(0, 1  , 5));
    try expectEqual(  0, density(0, 1.1, 5));

    try expectApproxEqRel(0                  , density(0, 3, 5), eps);
    try expectApproxEqRel(0.41015625         , density(1, 3, 5), eps);
    try expectApproxEqRel(0.06401463191586648, density(2, 3, 5), eps);
}

test "betaPrime.probability" {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                 , probability(0, 3, 5), eps);
    try expectApproxEqRel(0.7734375         , probability(1, 3, 5), eps);
    try expectApproxEqRel(0.9547325102880658, probability(2, 3, 5), eps);
}

test "betaPrime.quantile" {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.2958847929875766, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.4718562623302689, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(0.6926635008537015, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(1.0693555697769304, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf               , quantile(1  , 3, 5)     );
}

test "betaPrime.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.fedfe5d36ee54p-2, random.single(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.5ffd299be7e43p-2, random.single(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.28add14984721p-1, random.single(gen, 3, 5), eps);
}
