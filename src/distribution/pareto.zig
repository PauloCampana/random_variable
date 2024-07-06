//! Support: [k,∞)
//!
//! Parameters:
//! - α: `shape`   ∈ (0,∞)
//! - k: `minimum` ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = αk^α / x^(α + 1)
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

/// F(q) = 1 - (k / q)^α
pub fn probability(q: f64, shape: f64, minimum: f64) f64 {
    assert(isFinite(shape) and isFinite(minimum));
    assert(shape > 0 and minimum > 0);
    assert(!isNan(q));
    if (q < minimum) {
        return 0;
    }
    return 1 - std.math.pow(f64, minimum / q, shape);
}

/// S(t) = (k / t)^α
pub fn survival(t: f64, shape: f64, minimum: f64) f64 {
    assert(isFinite(shape) and isFinite(minimum));
    assert(shape > 0 and minimum > 0);
    assert(!isNan(t));
    if (t < minimum) {
        return 1;
    }
    return std.math.pow(f64, minimum / t, shape);
}

/// Q(p) = k / (1 - p)^(1 / α)
pub fn quantile(p: f64, shape: f64, minimum: f64) f64 {
    assert(isFinite(shape) and isFinite(minimum));
    assert(shape > 0 and minimum > 0);
    assert(0 <= p and p <= 1);
    return minimum * std.math.pow(f64, 1 - p, -1 / shape);
}

pub fn random(generator: std.Random, shape: f64, minimum: f64) f64 {
    assert(isFinite(shape) and isFinite(minimum));
    assert(shape > 0 and minimum > 0);
    const exp = generator.floatExp(f64);
    return minimum * @exp(exp / shape);
}

pub fn fill(buffer: []f64, generator: std.Random, shape: f64, minimum: f64) void {
    assert(isFinite(shape) and isFinite(minimum));
    assert(shape > 0 and minimum > 0);
    for (buffer) |*x| {
        const exp = generator.floatExp(f64);
        x.* = minimum * @exp(exp / shape);
    }
}

export fn rv_pareto_density(x: f64, shape: f64, minimum: f64) f64 {
    return density(x, shape, minimum);
}
export fn rv_pareto_probability(q: f64, shape: f64, minimum: f64) f64 {
    return probability(q, shape, minimum);
}
export fn rv_pareto_survival(t: f64, shape: f64, minimum: f64) f64 {
    return survival(t, shape, minimum);
}
export fn rv_pareto_quantile(p: f64, shape: f64, minimum: f64) f64 {
    return quantile(p, shape, minimum);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectApproxEqRel(0.6               , density(5, 3, 5), eps);
    try expectApproxEqRel(0.2893518518518518, density(6, 3, 5), eps);
    try expectApproxEqRel(0.1561849229487713, density(7, 3, 5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                 , probability(5, 3, 5), eps);
    try expectApproxEqRel(0.4212962962962962, probability(6, 3, 5), eps);
    try expectApproxEqRel(0.6355685131195335, probability(7, 3, 5), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 3, 5));
    try expectEqual(0, survival( inf, 3, 5));

    try expectApproxEqRel(1                 , survival(5, 3, 5), eps);
    try expectApproxEqRel(0.5787037037037037, survival(6, 3, 5), eps);
    try expectApproxEqRel(0.3644314868804664, survival(7, 3, 5), eps);
}

test quantile {
    try expectApproxEqRel(5                , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(5.386086725079709, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(5.928155507483438, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(6.786044041487266, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(8.549879733383484, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf              , quantile(1  , 3, 5)     );
}
