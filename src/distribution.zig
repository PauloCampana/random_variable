//! Cumulative distribution functions F(x)
//! for common probability distributions
//!
//! F(x) = P(X <= x)
//!
//! x ∈ [-∞, ∞]
//!
//! F(x) ∈ [0, 1]
//!
//! asserts invalid distribution parameters on Debug and ReleaseSafe

const std = @import("std");
const stdprob = @import("thirdyparty/prob.zig");

const assert = std.debug.assert;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const isFinite = std.math.isFinite; // tests false for both inf and nan

const testing = std.testing;
const epsilon = std.math.floatEps(f64);
const inf = std.math.inf(f64);

/// Cumulative distribution function of Uniform distribution
///
/// min and max ∈ (-∞, ∞)
pub fn uniform(q: f64, min: f64, max: f64) f64 {
    assert(isFinite(min));
    assert(isFinite(max));
    assert(!isNan(q));
    if (q <= min) {
        return 0;
    }
    if (q >= max) {
        return 1;
    }
    return (q - min) / (max - min);
}

test "distribution.uinform" {
    try testing.expectEqual(uniform(-inf, 0, 1), 0);
    try testing.expectEqual(uniform(-999, 0, 1), 0);
    try testing.expectEqual(uniform( 999, 0, 1), 1);
    try testing.expectEqual(uniform( inf, 0, 1), 1);

    try testing.expectApproxEqRel(uniform(3, 3, 5), 0  , epsilon);
    try testing.expectApproxEqRel(uniform(4, 3, 5), 0.5, epsilon);
    try testing.expectApproxEqRel(uniform(5, 3, 5), 1  , epsilon);
}

/// Cumulative distribution function of Bernoulli distribution
///
/// prob ∈ [0, 1]
pub fn bernoulli(q: f64, prob: f64) f64 {
    assert(isFinite(prob));
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

test "distribution.bernoulli" {
    try testing.expectEqual(bernoulli(-inf, 0.2), 0);
    try testing.expectEqual(bernoulli(-999, 0.2), 0);
    try testing.expectEqual(bernoulli( 999, 0.2), 1);
    try testing.expectEqual(bernoulli( inf, 0.2), 1);

    try testing.expectApproxEqRel(bernoulli(0  , 0.2), 0.8, epsilon);
    try testing.expectApproxEqRel(bernoulli(0.5, 0.2), 0.8, epsilon);
    try testing.expectApproxEqRel(bernoulli(1  , 0.2), 1  , epsilon);
}

/// Cumulative distribution function of Geometric distribution
///
/// prob ∈ [0, 1]
pub fn geometric(q: f64, prob: f64) !f64 {
    assert(isFinite(prob));
    assert(0 < prob and prob <= 1);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    const p = (@floor(q) + 1) * std.math.log1p(-prob);
    return -std.math.expm1(p);
}

test "distribution.geometric" {

}

// pub fn poisson(q: f64, lambda: f64) !f64 {
//     if (!isFinite(lambda) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (lambda < 0) {
//         return error.NegativeParam;
//     }
//     if (lambda == 0) {
//         return 1;
//     }
//     if (q < 0) {
//         return 0;
//     }
//     return 1 - try gamma(lambda, @floor(q) + 1, 1);
// }

// pub fn binomial(q: f64, size: u64, prob: f64) !f64 {
//     if (!isFinite(prob) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (prob < 0 or prob > 1) {
//         return error.ProbOutside01;
//     }
//     if (q < 0) {
//         return 0;
//     }
//     const fsize = @as(f64, @floatFromInt(size));
//     if (q >= fsize) {
//         return 1;
//     }
//     if (prob == 0) {
//         return 1;
//     }
//     if (prob == 1) {
//         return 0;
//     }
//     const fq = @floor(q);
//     return beta(prob, fsize - fq, fq + 1);
// }

// pub fn negativeBinomial(q: f64, size: u64, prob: f64) !f64 {
//     if (!isFinite(prob) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (prob < 0 or prob > 1) {
//         return error.ProbOutside01;
//     }
//     if (size == 0) {
//         return error.ZeroParam;
//     }
//     if (prob == 0) {
//         return error.ZeroParam;
//     }
//     if (q < 0) {
//         return 0;
//     }
//     const fsize = @as(f64, @floatFromInt(size));
//     return beta(prob, fsize, @floor(q) + 1);
// }

// pub fn exponential(q: f64, rate: f64) !f64 {
//     if (!isFinite(rate) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (rate < 0) {
//         return error.NegativeParam;
//     }
//     if (rate == 0) {
//         return error.ZeroParam;
//     }
//     if (q <= 0) {
//         return 0;
//     }
//     return -std.math.expm1(-rate * q);
// }

// pub fn weibull(q: f64, shape: f64, rate: f64) !f64 {
//     if (!isFinite(shape) or !isFinite(rate) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (shape < 0 or rate < 0) {
//         return error.NegativeParam;
//     }
//     if (shape == 0 or shape == 0) {
//         return error.ZeroParam;
//     }
//     if (q <= 0) {
//         return 0;
//     }
//     return -std.math.expm1(-std.math.pow(f64, rate * q, shape));
// }

// pub fn cauchy(q: f64, location: f64, scale: f64) !f64 {
//     if (!isFinite(location) or !isFinite(scale) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (scale < 0) {
//         return error.NegativeParam;
//     }
//     if (scale == 0) {
//         return error.ZeroParam;
//     }
//     const z = (q - location) / scale;
//     return 0.5 + std.math.atan(z) / std.math.pi;
// }

// pub fn logistic(q: f64, location: f64, scale: f64) !f64 {
//     if (!isFinite(location) or !isFinite(scale) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (scale < 0) {
//         return error.NegativeParam;
//     }
//     if (scale == 0) {
//         return error.ZeroParam;
//     }
//     const z = (q - location) / scale;
//     return 1 / (1 + @exp(-z));
// }

// pub fn gamma(q: f64, shape: f64, rate: f64) !f64 {
//     if (!isFinite(shape) or !isFinite(rate) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (shape < 0 or rate < 0) {
//         return error.NegativeParam;
//     }
//     if (shape == 0 or rate == 0) {
//         return error.ZeroParam;
//     }
//     if (q <= 0) {
//         return 0;
//     }
//     const z = rate * q;
//     return stdprob.incompleteGamma(shape, z);
// }

// pub fn chiSquared(q: f64, df: f64) !f64 {
//     return gamma(q, 0.5 * df, 0.5);
// }

// pub fn F(q: f64, df1: f64, df2: f64) !f64 {
//     const z = df1 * q / (df2 + df1 * q);
//     return beta(z, 0.5 * df1, 0.5 * df2);
// }

// pub fn beta(q: f64, shape1: f64, shape2: f64) !f64 {
//     if (!isFinite(shape1) or !isFinite(shape2) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (shape1 < 0 or shape2 < 0) {
//         return error.NegativeParam;
//     }
//     if (shape1 == 0 or shape2 == 0) {
//         return error.ZeroParam;
//     }
//     if (q <= 0) {
//         return 0;
//     }
//     if (q >= 1) {
//         return 1;
//     }
//     return stdprob.incompleteBeta(shape1, shape2, q);
// }

// pub fn normal(q: f64, mean: f64, sd: f64) !f64 {
//     if (!isFinite(mean) or !isFinite(sd) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (sd < 0) {
//         return error.NegativeParam;
//     }
//     if (sd == 0) {
//         return error.ZeroParam;
//     }
//     const z = (q - mean) / sd;
//     return stdprob.normalDist(z);
// }

// pub fn lognormal(q: f64, meanlog: f64, sdlog: f64) !f64 {
//     if (!isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (q <= 0) {
//         return 0;
//     }
//     return normal(@log(q), meanlog, sdlog);
// }

// pub fn t(q: f64, df: f64) !f64 {
//     if (!isFinite(df) or !isFinite(q)) {
//         return error.NonFiniteParam;
//     }
//     if (df < 0) {
//         return error.NegativeParam;
//     }
//     if (df == 0) {
//         return error.ZeroParam;
//     }
//     if (q <= 0) {
//         const z = df / (df + q * q);
//         return 0.5 * try beta(z, 0.5 * df, 0.5);
//     } else {
//         const z = q * q / (df + q * q);
//         return 0.5 * try beta(z, 0.5, 0.5 * df) + 0.5;
//     }
// }
