//! Probability density/mass functions f(x)/P(x)
//! for common probability distributions
//!
//! asserts invalid distribution parameters on Debug and ReleaseSafe

const std = @import("std");
const lnGamma = @import("thirdyparty/prob.zig").lnGamma;

const assert = std.debug.assert;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const isFinite = std.math.isFinite; // tests false for both inf and nan

const testing = std.testing;
const epsilon = std.math.floatEps(f64);
const inf = std.math.inf(f64);

/// Probability density function of Uniform distribution
///
/// min and max ∈ (-∞, ∞)
pub fn uniform(x: f64, min: f64, max: f64) f64 {
    assert(isFinite(min));
    assert(isFinite(max));
    assert(!isNan(x));
    if (x < min or x > max) {
        return 0;
    }
    return 1 / (max - min);
}

test "density.uniform" {
    try testing.expectEqual(uniform(-inf, 0, 1), 0);
    try testing.expectEqual(uniform(-999, 0, 1), 0);
    try testing.expectEqual(uniform( 999, 0, 1), 0);
    try testing.expectEqual(uniform( inf, 0, 1), 0);

    try testing.expectApproxEqRel(uniform(3, 3, 5), 0.5, epsilon);
    try testing.expectApproxEqRel(uniform(4, 3, 5), 0.5, epsilon);
    try testing.expectApproxEqRel(uniform(5, 3, 5), 0.5, epsilon);
}

/// Probability mass function of Bernoulli distribution
///
/// prob ∈ [0, 1]
pub fn bernoulli(x: u64, prob: f64) f64 {
    assert(isFinite(prob));
    assert(0 <= prob and prob <= 1);
    assert(!isNan(x));
    if (x == 0) {
        return 1 - prob;
    }
    if (x == 1) {
        return prob;
    }
    return 0;
}

test "density.bernoulli" {
    try testing.expectApproxEqRel(bernoulli(0  , 0.2), 0.8, epsilon);
    try testing.expectApproxEqRel(bernoulli(1  , 0.2), 0.2, epsilon);
    try testing.expectApproxEqRel(bernoulli(2  , 0.2), 0  , epsilon);
    try testing.expectApproxEqRel(bernoulli(999, 0.2), 0  , epsilon);
}

/// Probability mass function of Geometric distribution
///
/// prob ∈ (0, 1]
pub fn geometric(x: u64, prob: f64) f64 {
    assert(isFinite(prob));
    assert(0 < prob and prob <= 1);
    assert(!isNan(x));
    const fx = @as(f64, @floatFromInt(x));
    return prob * std.math.pow(f64, (1 - prob), fx);
}

test "density.geometric" {
    try testing.expectApproxEqRel(geometric(0  , 0.2), 0.2  , epsilon);
    try testing.expectApproxEqRel(geometric(1  , 0.2), 0.16 , epsilon);
    try testing.expectApproxEqRel(geometric(2  , 0.2), 0.128, epsilon);
    try testing.expectApproxEqAbs(geometric(999, 0.2), 0    , epsilon);
}

// /// Probability mass function of Poisson distribution
// ///
// /// lambda ∈ [0, ∞]
// pub fn poisson(x: u64, lambda: f64) !f64 {
//     assert(!isNan(lambda));
//     assert(lambda >= 0);
//     if (lambda == 0) {
//         return if (x == 0) 1 else 0;
//     }
//     if (isInf(lambda)) {
//         return 0;
//     }
//     const fx = @as(f64, @floatFromInt(x));
//     const log = -lambda + fx * @log(lambda) - lnGamma(fx + 1);
//     return @exp(log);
// }

// pub fn binomial(x: u64, size: u64, prob: f64) !f64 {
//     if (!isFinite(prob)) {
//         return error.NonFiniteParam;
//     }
//     if (prob < 0 or prob > 1) {
//         return error.ProbOutside01;
//     }
//     if (x > size) {
//         return 0;
//     }
//     if (prob == 0) {
//         return if (x == 0) 1 else 0;
//     }
//     if (prob == 1) {
//         return if (x == size) 1 else 0;
//     }
//     const fx = @as(f64, @floatFromInt(x));
//     const fsize = @as(f64, @floatFromInt(size));
//     const diff = fsize - fx;
//     const binom = lnGamma(fsize + 1) - lnGamma(fx + 1) - lnGamma(diff + 1);
//     const log = binom + fx * @log(prob) + diff * @log(1 - prob);
//     return @exp(log);
// }

// pub fn negativeBinomial(x: u64, size: u64, prob: f64) !f64 {
//     if (!isFinite(prob)) {
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
//     if (prob == 1) {
//         return if (x == 0) 1 else 0;
//     }
//     const fx = @as(f64, @floatFromInt(x));
//     const fsize = @as(f64, @floatFromInt(size));
//     const binom = lnGamma(fsize + fx) - lnGamma(fsize) - lnGamma(fx + 1);
//     const log = binom + fsize * @log(prob) + fx * @log(1 - prob);
//     return @exp(log);
// }

// pub fn exponential(x: f64, rate: f64) !f64 {
//     if (!isFinite(rate) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (rate < 0) {
//         return error.NegativeParam;
//     }
//     if (rate == 0) {
//         return error.ZeroParam;
//     }
//     if (x < 0) {
//         return 0;
//     }
//     return rate * @exp(-rate * x);
// }

// pub fn weibull(x: f64, shape: f64, rate: f64) !f64 {
//     if (!isFinite(shape) or !isFinite(rate) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (shape < 0 or rate < 0) {
//         return error.NegativeParam;
//     }
//     if (shape == 0 or shape == 0) {
//         return error.ZeroParam;
//     }
//     if (x <= 0) {
//         return 0;
//     }
//     const tmp1 = std.math.pow(f64, rate * x, shape - 1);
//     const tmp2 = tmp1 * rate * x;
//     return shape * rate * @exp(-tmp2) * tmp1;
// }

// pub fn cauchy(x: f64, location: f64, scale: f64) !f64 {
//     if (!isFinite(location) or !isFinite(scale) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (scale < 0) {
//         return error.NegativeParam;
//     }
//     if (scale == 0) {
//         return error.ZeroParam;
//     }
//     const z = (x - location) / scale;
//     return 1 / (std.math.pi * scale * (1 + z * z));
// }

// pub fn logistic(x: f64, location: f64, scale: f64) !f64 {
//     if (!isFinite(location) or !isFinite(scale) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (scale < 0) {
//         return error.NegativeParam;
//     }
//     if (scale == 0) {
//         return error.ZeroParam;
//     }
//     const z = @abs(x - location) / scale;
//     const tmp1 = @exp(-z);
//     const tmp2 = tmp1 + 1;
//     return tmp1 / (scale * tmp2 * tmp2);
// }

// pub fn gamma(x: f64, shape: f64, rate: f64) !f64 {
//     if (!isFinite(shape) or !isFinite(rate) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (shape < 0 or rate < 0) {
//         return error.NegativeParam;
//     }
//     if (shape == 0 or rate == 0) {
//         return error.ZeroParam;
//     }
//     if (x < 0) {
//         return 0;
//     }
//     const z = rate * x;
//     const den = lnGamma(shape) + @log(x);
//     const num = shape * @log(z) - z;
//     return @exp(num - den);
// }

// pub fn chiSquared(x: f64, df: f64) !f64 {
//     return gamma(x, 0.5 * df, 0.5);
// }

// pub fn F(x: f64, df1: f64, df2: f64) !f64 {
//     if (!isFinite(df1) or !isFinite(df2) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (df1 < 0 or df2 < 0) {
//         return error.NegativeParam;
//     }
//     if (df1 == 0 or df2 == 0) {
//         return error.ZeroParam;
//     }
//     if (x < 0) {
//         return 0;
//     }
//     const df3 = df1 / 2;
//     const df4 = df2 / 2;
//     const df5 = df3 + df4;
//     const num1 = df3 * @log(df1) + df4 * @log(df2) + (df3 - 1) * @log(x);
//     const num2 = -df5 * @log(df2 + df1 * x);
//     const den = lnGamma(df3) + lnGamma(df4) - lnGamma(df5);
//     return @exp(num1 + num2 - den);
// }

// pub fn beta(x: f64, shape1: f64, shape2: f64) !f64 {
//     if (!isFinite(shape1) or !isFinite(shape2) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (shape1 < 0 or shape2 < 0) {
//         return error.NegativeParam;
//     }
//     if (shape1 == 0 or shape2 == 0) {
//         return error.ZeroParam;
//     }
//     if (x < 0 or x > 1) {
//         return 0;
//     }
//     const num = (shape1 - 1) * @log(x) + (shape2 - 1) * std.math.log1p(-x);
//     const den = lnGamma(shape1) + lnGamma(shape2) - lnGamma(shape1 + shape2);
//     return @exp(num - den);
// }

// pub fn normal(x: f64, mean: f64, sd: f64) !f64 {
//     if (!isFinite(mean) or !isFinite(sd) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (sd < 0) {
//         return error.NegativeParam;
//     }
//     if (sd == 0) {
//         return error.ZeroParam;
//     }
//     const z = (x - mean) / sd;
//     const sqrt2pi = @sqrt(2 * std.math.pi);
//     return @exp(-0.5 * z * z) / (sd * sqrt2pi);
// }

// pub fn lognormal(x: f64, meanlog: f64, sdlog: f64) !f64 {
//     if (!isFinite(meanlog) or !isFinite(sdlog) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (sdlog < 0) {
//         return error.NegativeParam;
//     }
//     if (sdlog == 0) {
//         return error.ZeroParam;
//     }
//     if (x <= 0) {
//         return 0;
//     }
//     const z = (@log(x) - meanlog) / sdlog;
//     const sqrt2pi = @sqrt(2 * std.math.pi);
//     return @exp(-0.5 * z * z) / (x * sdlog * sqrt2pi);
// }

// pub fn t(x: f64, df: f64) !f64 {
//     if (!isFinite(df) or !isFinite(x)) {
//         return error.NonFiniteParam;
//     }
//     if (df < 0) {
//         return error.NegativeParam;
//     }
//     if (df == 0) {
//         return error.ZeroParam;
//     }
//     const num = (df + 1) / 2 * @log(df / (df + x * x)) - 0.5 * @log(df);
//     const den = lnGamma(df / 2) + lnGamma(0.5) - lnGamma((df + 1) / 2);
//     return @exp(num - den);
// }
