//! Probability functions F(q),
//! also known as distribution or cumulative distribution.
//!
//! Represents the probability of a random variable X
//! being less than or equal to a certain value: F(q) = P(X <= q).
//!
//! Maps every number into a probability, F(-∞) = 0 and F(∞) = 1.

const std = @import("std");
const stdprob = @import("../thirdyparty/prob.zig");

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15
const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

/// min and max ∈ (-∞,∞)
pub fn uniform(q: f64, min: f64, max: f64) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(!isNan(q));
    if (q <= min) {
        return 0;
    }
    if (q >= max) {
        return 1;
    }
    return (q - min) / (max - min);
}

test "probability.uniform" {
    try expectEqual(@as(f64, 0), uniform(-inf, 0, 1));
    try expectEqual(@as(f64, 1), uniform( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0  ), uniform(3  , 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.2), uniform(3.4, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.4), uniform(3.8, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.6), uniform(4.2, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.8), uniform(4.6, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 1  ), uniform(5  , 3, 5), eps);
}

/// prob ∈ [0,1]
pub fn bernoulli(q: f64, prob: f64) f64 {
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

test "probability.bernoulli" {
    try expectEqual(@as(f64, 0), bernoulli(-inf, 0.2));
    try expectEqual(@as(f64, 1), bernoulli( inf, 0.2));

    try expectApproxEqRel(@as(f64, 0  ), bernoulli(-0.1, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.8), bernoulli( 0  , 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.8), bernoulli( 0.1, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.8), bernoulli( 0.9, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1  ), bernoulli( 1  , 0.2), eps);
    try expectApproxEqRel(@as(f64, 1  ), bernoulli( 1.1, 0.2), eps);
}

/// prob ∈ (0,1]
pub fn geometric(q: f64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    const p = (@floor(q) + 1) * std.math.log1p(-prob);
    return -std.math.expm1(p);
}

test "probability.geometric" {
    try expectEqual(@as(f64, 0), geometric(-inf, 0.2));
    try expectEqual(@as(f64, 1), geometric( inf, 0.2));

    try expectApproxEqRel(@as(f64, 0   ), geometric(-0.1, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.2 ), geometric( 0  , 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.2 ), geometric( 0.1, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.2 ), geometric( 0.9, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.36), geometric( 1  , 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.36), geometric( 1.1, 0.2), eps);
}

/// lambda ∈ (0,∞)
pub fn poisson(q: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    if (isInf(q)) {
        return 1;
    }
    return 1 - stdprob.incompleteGamma(@floor(q) + 1, lambda);
}

test "probability.poisson" {
    try expectEqual(@as(f64, 0), poisson(-inf, 3));
    try expectEqual(@as(f64, 1), poisson( inf, 3));

    try expectApproxEqRel(@as(f64, 0                 ), poisson(-0.1, 3), eps);
    try expectApproxEqRel(@as(f64, 0.0497870683678639), poisson( 0  , 3), eps);
    try expectApproxEqRel(@as(f64, 0.0497870683678639), poisson( 0.1, 3), eps);
    try expectApproxEqRel(@as(f64, 0.0497870683678639), poisson( 0.9, 3), eps);
    try expectApproxEqRel(@as(f64, 0.1991482734714558), poisson( 1  , 3), eps);
    try expectApproxEqRel(@as(f64, 0.1991482734714558), poisson( 1.1, 3), eps);
}

/// size ∈ {0,1,2,⋯}, prob ∈ [0,1]
pub fn binomial(q: f64, size: u64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(!isNan(q));
    const fsize = @as(f64, @floatFromInt(size));
    if (q < 0) {
        return 0;
    }
    if (q >= fsize) {
        return 1;
    }
    if (prob == 0) {
        return 1;
    }
    if (prob == 1) {
        return 0;
    }
    const fq = @floor(q);
    return stdprob.incompleteBeta(fsize - fq, fq + 1, 1 - prob);
}

test "probability.binomial" {
    try expectEqual(@as(f64, 0), binomial(-inf, 10, 0.2));
    try expectEqual(@as(f64, 1), binomial( inf, 10, 0.2));

    try expectEqual(@as(f64, 1), binomial( 0,  0, 0.2));
    try expectEqual(@as(f64, 1), binomial( 1,  0, 0.2));
    try expectEqual(@as(f64, 1), binomial( 0, 10, 0  ));
    try expectEqual(@as(f64, 1), binomial( 1, 10, 0  ));
    try expectEqual(@as(f64, 0), binomial( 9, 10, 1  ));
    try expectEqual(@as(f64, 1), binomial(10, 10, 1  ));
    try expectEqual(@as(f64, 1), binomial(11, 10, 1  ));

    try expectApproxEqRel(@as(f64, 0           ), binomial(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.1073741824), binomial( 0  , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.1073741824), binomial( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.1073741824), binomial( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.3758096384), binomial( 1  , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.3758096384), binomial( 1.1, 10, 0.2), eps);
}

/// size ∈ {1,2,3,⋯}, prob ∈ (0,1]
pub fn negativeBinomial(q: f64, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    if (isInf(q) or prob == 1) {
        return 1;
    }
    const fsize = @as(f64, @floatFromInt(size));
    return stdprob.incompleteBeta(fsize, @floor(q) + 1, prob);
}

test "probability.negativeBinomial" {
    try expectEqual(@as(f64, 0), negativeBinomial(-inf, 10, 0.2));
    try expectEqual(@as(f64, 1), negativeBinomial( inf, 10, 0.2));

    try expectEqual(@as(f64, 1), negativeBinomial( 0, 10, 1  ));
    try expectEqual(@as(f64, 1), negativeBinomial( 1, 10, 1  ));

    try expectApproxEqRel(@as(f64, 0           ), negativeBinomial(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.0000001024), negativeBinomial( 0  , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.0000001024), negativeBinomial( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.0000001024), negativeBinomial( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.0000009216), negativeBinomial( 1  , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.0000009216), negativeBinomial( 1.1, 10, 0.2), eps);
}

/// rate ∈ (0,∞)
pub fn exponential(q: f64, rate: f64) f64 {
    assert(isFinite(rate));
    assert(rate > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = rate * q;
    return -std.math.expm1(-z);
}

test "probability.exponential" {
    try expectEqual(@as(f64, 0), exponential(-inf, 3));
    try expectEqual(@as(f64, 1), exponential( inf, 3));

    try expectApproxEqRel(@as(f64, 0                 ), exponential(0, 3), eps);
    try expectApproxEqRel(@as(f64, 0.9502129316321360), exponential(1, 3), eps);
    try expectApproxEqRel(@as(f64, 0.9975212478233336), exponential(2, 3), eps);
}

/// shape and rate ∈ (0,∞)
pub fn weibull(q: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = rate * q;
    const za = std.math.pow(f64, z, shape);
    return -std.math.expm1(-za);
}

test "probability.weibull" {
    try expectEqual(@as(f64, 0), weibull(-inf, 3, 0.5));
    try expectEqual(@as(f64, 1), weibull( inf, 3, 0.5));

    try expectApproxEqRel(@as(f64, 0                 ), weibull(0, 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 0.1175030974154046), weibull(1, 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 0.6321205588285577), weibull(2, 3, 0.5), eps);
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn cauchy(q: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(q));
    const z = (q - location) / scale;
    return 0.5 + std.math.atan(z) / std.math.pi;
}

test "probability.cauchy" {
    try expectEqual(@as(f64, 0), cauchy(-inf, 0, 1));
    try expectEqual(@as(f64, 1), cauchy( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0.5               ), cauchy(0, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.75              ), cauchy(1, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.8524163823495667), cauchy(2, 0, 1), eps);
}

/// location ∈ (-∞,∞)
///
/// scale ∈ (0,∞)
pub fn logistic(q: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(q));
    const z = (q - location) / scale;
    return 1 / (1 + @exp(-z));
}

test "probability.logistic" {
    try expectEqual(@as(f64, 0), logistic(-inf, 0, 1));
    try expectEqual(@as(f64, 1), logistic( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0.5               ), logistic(0, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.7310585786300049), logistic(1, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.8807970779778823), logistic(2, 0, 1), eps);
}

/// shape and rate ∈ (0,∞)
pub fn gamma(q: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = rate * q;
    return stdprob.incompleteGamma(shape, z);
}

test "probability.gamma" {
    try expectEqual(@as(f64, 0), gamma(-inf, 3, 5));
    try expectEqual(@as(f64, 1), gamma( inf, 3, 5));

    try expectApproxEqRel(@as(f64, 0                 ), gamma(0, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.8753479805169189), gamma(1, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.9972306042844884), gamma(2, 3, 5), eps);
}

/// df ∈ (0,∞)
pub fn chiSquared(q: f64, df: f64) f64 {
    return gamma(q, 0.5 * df, 0.5);
}

test "probability.chiSquared" {
    try expectEqual(@as(f64, 0), chiSquared(-inf, 3));
    try expectEqual(@as(f64, 1), chiSquared( inf, 3));

    try expectApproxEqRel(@as(f64, 0                 ), chiSquared(0, 3), eps);
    try expectApproxEqRel(@as(f64, 0.1987480430987992), chiSquared(1, 3), eps);
    try expectApproxEqRel(@as(f64, 0.4275932955291208), chiSquared(2, 3), eps);
}

/// df1 and df2 ∈ (0,∞)
pub fn F(q: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    if (isInf(q)) {
        return 1;
    }
    const z = df1 * q;
    const p = z / (df2 + z);
    return stdprob.incompleteBeta(0.5 * df1, 0.5 * df2, p);
}

test "probability.F" {
    try expectEqual(@as(f64, 0), F(-inf, 3, 5));
    try expectEqual(@as(f64, 1), F( inf, 3, 5));

    try expectApproxEqRel(@as(f64, 0                 ), F(0, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.5351452100063649), F(1, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.7673760819999214), F(2, 3, 5), eps);
}

/// shape1 and shape2 ∈ (0,∞)
pub fn beta(q: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    if (q >= 1) {
        return 1;
    }
    return stdprob.incompleteBeta(shape1, shape2, q);
}

test "probability.beta" {
    try expectEqual(@as(f64, 0), beta(-inf, 3, 5));
    try expectEqual(@as(f64, 1), beta( inf, 3, 5));

    try expectApproxEqRel(@as(f64, 0       ), beta(0  , 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.148032), beta(0.2, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.995328), beta(0.8, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 1       ), beta(1  , 3, 5), eps);
}

/// mean ∈ (-∞,∞)
///
/// sd ∈ (0,∞)
pub fn normal(q: f64, mean: f64, sd: f64) f64 {
    assert(isFinite(mean) and isFinite(sd));
    assert(sd > 0);
    assert(!isNan(q));
    const z = (q - mean) / sd;
    return stdprob.normalDist(z);
}

test "probability.normal" {
    try expectEqual(@as(f64, 0), normal(-inf, 0, 1));
    try expectEqual(@as(f64, 1), normal( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0.5               ), normal(0, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.8413447460685429), normal(1, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.9772498680518208), normal(2, 0, 1), eps);
}

/// meanlog ∈ (-∞,∞)
///
/// sdlog ∈ (0,∞)
pub fn logNormal(q: f64, meanlog: f64, sdlog: f64) f64 {
    assert(isFinite(meanlog) and isFinite(sdlog));
    assert(sdlog > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = (@log(q) - meanlog) / sdlog;
    return stdprob.normalDist(z);
}

test "probability.logNormal" {
    try expectEqual(@as(f64, 0), logNormal(-inf, 0, 1));
    try expectEqual(@as(f64, 1), logNormal( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0                 ), logNormal(0, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.5               ), logNormal(1, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.7558914042144173), logNormal(2, 0, 1), eps);
}

/// df ∈ (0,∞)
pub fn t(q: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(!isNan(q));
    if (std.math.isInf(q)) {
        return if (q < 0) 0 else 1;
    }
    const z = q * q;
    if (q < 0) {
        const p = df / (df + z);
        return 0.5 * stdprob.incompleteBeta(0.5 * df, 0.5, p);
    } else {
        const p = z / (df + z);
        return 0.5 * stdprob.incompleteBeta(0.5, 0.5 * df, p) + 0.5;
    }
}

test "probability.t" {
    try expectEqual(@as(f64, 0), t(-inf, 3));
    try expectEqual(@as(f64, 1), t( inf, 3));

    try expectApproxEqRel(@as(f64, 0.5               ), t(0, 3), eps);
    try expectApproxEqRel(@as(f64, 0.8044988905221148), t(1, 3), eps);
    try expectApproxEqRel(@as(f64, 0.9303370157205784), t(2, 3), eps);
}
