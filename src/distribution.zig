//! Cumulative distribution functions F(x)
//! for common probability distributions
//!
//! F(x) = P(X <= x)
//!
//! x ∈ [-∞,∞]
//!
//! F(x) ∈ [0,1]
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe
//! such as ±NaN, ±Inf, probabilities outside [0,1],
//! negative or zero shape, df, rate or scale parameters.

const std = @import("std");
const stdprob = @import("thirdyparty/prob.zig");
const expectEqual = @import("thirdyparty/testing.zig").expectEqual;
const expectApproxEqRel = @import("thirdyparty/testing.zig").expectApproxEqRel;

const assert = std.debug.assert;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const isFinite = std.math.isFinite; // tests false for both inf and nan
const inf = std.math.inf(f64);

/// Cumulative distribution function of Uniform distribution
///
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

test "distribution.uniform" {
    try expectEqual(uniform(-inf, 0, 1), 0);
    try expectEqual(uniform( inf, 0, 1), 1);

    try expectApproxEqRel(uniform(3  , 3, 5), 0  );
    try expectApproxEqRel(uniform(3.4, 3, 5), 0.2);
    try expectApproxEqRel(uniform(3.8, 3, 5), 0.4);
    try expectApproxEqRel(uniform(4.2, 3, 5), 0.6);
    try expectApproxEqRel(uniform(4.6, 3, 5), 0.8);
    try expectApproxEqRel(uniform(5  , 3, 5), 1  );
}

/// Cumulative distribution function of Bernoulli distribution
///
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

test "distribution.bernoulli" {
    try expectEqual(bernoulli(-inf, 0.2), 0);
    try expectEqual(bernoulli( inf, 0.2), 1);

    try expectApproxEqRel(bernoulli(-0.1, 0.2), 0  );
    try expectApproxEqRel(bernoulli( 0  , 0.2), 0.8);
    try expectApproxEqRel(bernoulli( 0.1, 0.2), 0.8);
    try expectApproxEqRel(bernoulli( 0.9, 0.2), 0.8);
    try expectApproxEqRel(bernoulli( 1  , 0.2), 1  );
    try expectApproxEqRel(bernoulli( 1.1, 0.2), 1  );
}

/// Cumulative distribution function of Geometric distribution
///
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

test "distribution.geometric" {
    try expectEqual(geometric(-inf, 0.2), 0);
    try expectEqual(geometric( inf, 0.2), 1);

    try expectApproxEqRel(geometric(-0.1, 0.2), 0   );
    try expectApproxEqRel(geometric( 0  , 0.2), 0.2 );
    try expectApproxEqRel(geometric( 0.1, 0.2), 0.2 );
    try expectApproxEqRel(geometric( 0.9, 0.2), 0.2 );
    try expectApproxEqRel(geometric( 1  , 0.2), 0.36);
    try expectApproxEqRel(geometric( 1.1, 0.2), 0.36);
}

/// Cumulative distribution function of Poisson distribution
///
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

test "distribution.poisson" {
    try expectEqual(poisson(-inf, 3), 0);
    try expectEqual(poisson( inf, 3), 1);

    try expectApproxEqRel(poisson(-0.1, 3), 0                 );
    try expectApproxEqRel(poisson( 0  , 3), 0.0497870683678639);
    try expectApproxEqRel(poisson( 0.1, 3), 0.0497870683678639);
    try expectApproxEqRel(poisson( 0.9, 3), 0.0497870683678639);
    try expectApproxEqRel(poisson( 1  , 3), 0.1991482734714558);
    try expectApproxEqRel(poisson( 1.1, 3), 0.1991482734714558);
}

/// Cumulative distribution function of Binomial distribution
///
/// size ∈ {0,1,2,⋯}
///
/// prob ∈ [0,1]
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

test "distribution.binomial" {
    try expectEqual(binomial(-inf, 10, 0.2), 0);
    try expectEqual(binomial( inf, 10, 0.2), 1);

    try expectEqual(binomial( 0,  0, 0.2), 1);
    try expectEqual(binomial( 1,  0, 0.2), 1);
    try expectEqual(binomial( 0, 10, 0  ), 1);
    try expectEqual(binomial( 1, 10, 0  ), 1);
    try expectEqual(binomial( 9, 10, 1  ), 0);
    try expectEqual(binomial(10, 10, 1  ), 1);
    try expectEqual(binomial(11, 10, 1  ), 1);

    try expectApproxEqRel(binomial(-0.1, 10, 0.2), 0           );
    try expectApproxEqRel(binomial( 0  , 10, 0.2), 0.1073741824);
    try expectApproxEqRel(binomial( 0.1, 10, 0.2), 0.1073741824);
    try expectApproxEqRel(binomial( 0.9, 10, 0.2), 0.1073741824);
    try expectApproxEqRel(binomial( 1  , 10, 0.2), 0.3758096384);
    try expectApproxEqRel(binomial( 1.1, 10, 0.2), 0.3758096384);
}

/// Cumulative distribution function of Negative Binomial distribution
///
/// size ∈ {1,2,3,⋯}
///
/// prob ∈ (0,1]
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

test "distribution.negativeBinomial" {
    try expectEqual(negativeBinomial(-inf, 10, 0.2), 0);
    try expectEqual(negativeBinomial( inf, 10, 0.2), 1);

    try expectEqual(negativeBinomial( 0, 10, 1  ), 1);
    try expectEqual(negativeBinomial( 1, 10, 1  ), 1);

    try expectApproxEqRel(negativeBinomial(-0.1, 10, 0.2), 0           );
    try expectApproxEqRel(negativeBinomial( 0  , 10, 0.2), 0.0000001024);
    try expectApproxEqRel(negativeBinomial( 0.1, 10, 0.2), 0.0000001024);
    try expectApproxEqRel(negativeBinomial( 0.9, 10, 0.2), 0.0000001024);
    try expectApproxEqRel(negativeBinomial( 1  , 10, 0.2), 0.0000009216);
    try expectApproxEqRel(negativeBinomial( 1.1, 10, 0.2), 0.0000009216);
}

/// Cumulative distribution function of Exponential distribution
///
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

test "distribution.exponential" {
    try expectEqual(exponential(-inf, 3), 0);
    try expectEqual(exponential( inf, 3), 1);

    try expectApproxEqRel(exponential(0, 3), 0                 );
    try expectApproxEqRel(exponential(1, 3), 0.9502129316321360);
    try expectApproxEqRel(exponential(2, 3), 0.9975212478233336);
}

/// Cumulative distribution function of Weibull distribution
///
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

test "distribution.weibull" {
    try expectEqual(weibull(-inf, 3, 0.5), 0);
    try expectEqual(weibull( inf, 3, 0.5), 1);

    try expectApproxEqRel(weibull(0, 3, 0.5), 0                 );
    try expectApproxEqRel(weibull(1, 3, 0.5), 0.1175030974154046);
    try expectApproxEqRel(weibull(2, 3, 0.5), 0.6321205588285577);
}

/// Cumulative distribution function of Cauchy distribution
///
/// location ∈ (-∞,∞)
///
/// scale ∈ (0,∞)
pub fn cauchy(q: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(q));
    const z = (q - location) / scale;
    return 0.5 + std.math.atan(z) / std.math.pi;
}

test "distribution.cauchy" {
    try expectEqual(cauchy(-inf, 0, 1), 0);
    try expectEqual(cauchy( inf, 0, 1), 1);

    try expectApproxEqRel(cauchy(0, 0, 1), 0.5               );
    try expectApproxEqRel(cauchy(1, 0, 1), 0.75              );
    try expectApproxEqRel(cauchy(2, 0, 1), 0.8524163823495667);
}

/// Cumulative distribution function of Logistic distribution
///
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

test "distribution.logistic" {
    try expectEqual(logistic(-inf, 0, 1), 0);
    try expectEqual(logistic( inf, 0, 1), 1);

    try expectApproxEqRel(logistic(0, 0, 1), 0.5               );
    try expectApproxEqRel(logistic(1, 0, 1), 0.7310585786300049);
    try expectApproxEqRel(logistic(2, 0, 1), 0.8807970779778823);
}

/// Cumulative distribution function of Gamma distribution
///
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

test "distribution.gamma" {
    try expectEqual(gamma(-inf, 3, 5), 0);
    try expectEqual(gamma( inf, 3, 5), 1);

    try expectApproxEqRel(gamma(0, 3, 5), 0                 );
    try expectApproxEqRel(gamma(1, 3, 5), 0.8753479805169189);
    try expectApproxEqRel(gamma(2, 3, 5), 0.9972306042844884);
}

/// Cumulative distribution function of Chi Squared distribution
///
/// df ∈ (0,∞)
pub fn chiSquared(q: f64, df: f64) f64 {
    return gamma(q, 0.5 * df, 0.5);
}

test "distribution.chiSquared" {
    try expectEqual(chiSquared(-inf, 3), 0);
    try expectEqual(chiSquared( inf, 3), 1);

    try expectApproxEqRel(chiSquared(0, 3), 0                 );
    try expectApproxEqRel(chiSquared(1, 3), 0.1987480430987992);
    try expectApproxEqRel(chiSquared(2, 3), 0.4275932955291208);
}

/// Cumulative distribution function of F distribution
///
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

test "distribution.F" {
    try expectEqual(F(-inf, 3, 5), 0);
    try expectEqual(F( inf, 3, 5), 1);

    try expectApproxEqRel(F(0, 3, 5), 0                 );
    try expectApproxEqRel(F(1, 3, 5), 0.5351452100063649);
    try expectApproxEqRel(F(2, 3, 5), 0.7673760819999214);
}

/// Cumulative distribution function of Beta distribution
///
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

test "distribution.beta" {
    try expectEqual(beta(-inf, 3, 5), 0);
    try expectEqual(beta( inf, 3, 5), 1);

    try expectApproxEqRel(beta(0  , 3, 5), 0       );
    try expectApproxEqRel(beta(0.2, 3, 5), 0.148032);
    try expectApproxEqRel(beta(0.8, 3, 5), 0.995328);
    try expectApproxEqRel(beta(1  , 3, 5), 1       );
}

/// Cumulative distribution function of Normal distribution
///
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

test "distribution.normal" {
    try expectEqual(normal(-inf, 0, 1), 0);
    try expectEqual(normal( inf, 0, 1), 1);

    try expectApproxEqRel(normal(0, 0, 1), 0.5               );
    try expectApproxEqRel(normal(1, 0, 1), 0.8413447460685429);
    try expectApproxEqRel(normal(2, 0, 1), 0.9772498680518208);
}

/// Cumulative distribution function of Log-normal distribution
///
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

test "distribution.logNormal" {
    try expectEqual(logNormal(-inf, 0, 1), 0);
    try expectEqual(logNormal( inf, 0, 1), 1);

    try expectApproxEqRel(logNormal(0, 0, 1), 0                 );
    try expectApproxEqRel(logNormal(1, 0, 1), 0.5               );
    try expectApproxEqRel(logNormal(2, 0, 1), 0.7558914042144173);
}

/// Cumulative distribution function of t distribution
///
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

test "density.t" {
    try expectEqual(t(-inf, 3), 0);
    try expectEqual(t( inf, 3), 1);

    try expectApproxEqRel(t(0, 3), 0.5               );
    try expectApproxEqRel(t(1, 3), 0.8044988905221148);
    try expectApproxEqRel(t(2, 3), 0.9303370157205784);
}
