//! Parameters:
//! - α: `shape` ∈ (0,∞)
//! - k: `minimum` ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const parameters = 2;

/// f(x) = αk^α / x^(α + 1).
pub fn density(x: f64, shape: f64, minimum: f64) f64 {
    assert(isFinite(shape) and isFinite(minimum));
    assert(shape > 0 and minimum > 0);
    assert(!isNan(x));
    if (x < minimum) {
        return 0;
    }
    const num = @log(shape) + shape * @log(minimum);
    const den = (shape + 1) * @log(x);
    return @exp(num - den);
}

/// F(q) = 1 - (k / q)^α.
pub fn probability(q: f64, shape: f64, minimum: f64) f64 {
    assert(isFinite(shape) and isFinite(minimum));
    assert(shape > 0 and minimum > 0);
    assert(!isNan(q));
    if (q < minimum) {
        return 0;
    }
    return 1 - std.math.pow(f64, minimum / q, shape);
}

/// Q(p) = k / (1 - p)^(1 / α).
pub fn quantile(p: f64, shape: f64, minimum: f64) f64 {
    assert(isFinite(shape) and isFinite(minimum));
    assert(shape > 0 and minimum > 0);
    assert(0 <= p and p <= 1);
    return minimum * std.math.pow(f64, 1 - p, -1 / shape);
}

/// Uses the quantile function.
pub const random = struct {
    fn implementation(generator: std.rand.Random, shape: f64, minimum: f64) f64 {
        const uni = generator.float(f64);
        return minimum * std.math.pow(f64, uni, -1 / shape);
    }

    pub fn single(generator: std.rand.Random, shape: f64, minimum: f64) f64 {
        assert(isFinite(shape) and isFinite(minimum));
        assert(shape > 0 and minimum > 0);
        return implementation(generator, shape, minimum);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, shape: f64, minimum: f64) []f64 {
        assert(isFinite(shape) and isFinite(minimum));
        assert(shape > 0 and minimum > 0);
        for (buf) |*x| {
            x.* = implementation(generator, shape, minimum);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, shape: f64, minimum: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, shape, minimum);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "pareto.density" {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectApproxEqRel(0.6               , density(5, 3, 5), eps);
    try expectApproxEqRel(0.2893518518518518, density(6, 3, 5), eps);
    try expectApproxEqRel(0.1561849229487713, density(7, 3, 5), eps);
}

test "pareto.probability" {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                 , probability(5, 3, 5), eps);
    try expectApproxEqRel(0.4212962962962962, probability(6, 3, 5), eps);
    try expectApproxEqRel(0.6355685131195335, probability(7, 3, 5), eps);
}

test "pareto.quantile" {
    try expectApproxEqRel(5                , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(5.386086725079709, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(5.928155507483438, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(6.786044041487266, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(8.549879733383484, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf              , quantile(1  , 3, 5)     );
}

test "pareto.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.bfbca14ce8587p2, random.implementation(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.adb0012df4ddep2, random.implementation(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.93b54a550f660p2, random.implementation(gen, 3, 5), eps);
}
