//! Quantile functions Q(x)
//! for common probability distributions
//!
//! Q(x) = F⁻¹(x)
//!
//! x ∈ [0,1]
//!
//! Q(x) ∈ [-∞,∞]
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe
//! such as ±NaN, ±Inf, probabilities outside [0,1],
//! negative or zero shape, df, rate or scale parameters.

const std = @import("std");
const stdprob = @import("thirdyparty/prob.zig");
const density = @import("density.zig");
const expectEqual = @import("thirdyparty/testing.zig").expectEqual;
const expectApproxEqRel = @import("thirdyparty/testing.zig").expectApproxEqRel;

const assert = std.debug.assert;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const isFinite = std.math.isFinite; // tests false for both inf and nan
const inf = std.math.inf(f64);

/// Quantile function of Uniform distribution
///
/// min and max ∈ (-∞,∞)
pub fn uniform(p: f64, min: f64, max: f64) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(0 <= p and p <= 1);
    return min + (max - min) * p;
}

test "quantile.uniform" {
    try expectApproxEqRel(uniform(0  , 3, 5), 3  );
    try expectApproxEqRel(uniform(0.2, 3, 5), 3.4);
    try expectApproxEqRel(uniform(0.4, 3, 5), 3.8);
    try expectApproxEqRel(uniform(0.6, 3, 5), 4.2);
    try expectApproxEqRel(uniform(0.8, 3, 5), 4.6);
    try expectApproxEqRel(uniform(1  , 3, 5), 5  );
}

/// Quantile function of Bernoulli distribution
///
/// prob ∈ [0,1]
pub fn bernoulli(p: f64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(0 <= p and p <= 1);
    return if (p > 1 - prob) 1 else 0;
}

test "quantile.bernoulli" {
    try expectApproxEqRel(bernoulli(0   , 0.2), 0);
    try expectApproxEqRel(bernoulli(0.79, 0.2), 0);
    try expectApproxEqRel(bernoulli(0.8 , 0.2), 0);
    try expectApproxEqRel(bernoulli(0.81, 0.2), 1);
    try expectApproxEqRel(bernoulli(1   , 0.2), 1);
}

/// Quantile function of Geometric distribution
///
/// prob ∈ (0,1]
pub fn geometric(p: f64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(0 <= p and p <= 1);
    if (p == 1) {
        return inf;
    }
    if (p <= prob) {
        return 0;
    }
    return @floor(std.math.log1p(-p) / std.math.log1p(-prob));
}

test "quantile.geometric" {
    try expectApproxEqRel(geometric(0   , 0.2), 0  );
    try expectApproxEqRel(geometric(0.19, 0.2), 0  );
    try expectApproxEqRel(geometric(0.2 , 0.2), 0  );
    try expectApproxEqRel(geometric(0.21, 0.2), 1  );
    try expectApproxEqRel(geometric(0.35, 0.2), 1  );
    try expectApproxEqRel(geometric(0.36, 0.2), 1  );
    try expectApproxEqRel(geometric(0.37, 0.2), 2  );
    try expectEqual      (geometric(1   , 0.2), inf);
}

/// Quantile function of Poisson distribution
///
/// lambda ∈ (0,∞)
pub fn poisson(p: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(0 <= p and p <= 1);
    if (p == 1) {
        return inf;
    }
    var q: f64 = 0;
    var sum: f64 = 0;
    while (true) : (q += 1) {
        sum += density.poisson(q, lambda);
        if (sum >= p) {
            return q;
        }
    }
}

test "quantile.poisson" {
    try expectApproxEqRel(poisson(0                 , 3), 0  );
    try expectApproxEqRel(poisson(0.0497870683678638, 3), 0  );
    try expectApproxEqRel(poisson(0.0497870683678639, 3), 0  );
    try expectApproxEqRel(poisson(0.0497870683678640, 3), 1  );
    try expectApproxEqRel(poisson(0.1991482734714556, 3), 1  );
    try expectApproxEqRel(poisson(0.1991482734714557, 3), 1  );
    try expectApproxEqRel(poisson(0.1991482734714559, 3), 2  );
    try expectEqual      (poisson(1                 , 3), inf);
}

/// Quantile function of Binomial distribution
///
/// size ∈ {0,1,2,⋯}
///
/// prob ∈ [0,1]
pub fn binomial(p: f64, size: u64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(0 <= p and p <= 1);
    const fsize = @as(f64, @floatFromInt(size));
    if (p == 0) {
        return 0;
    }
    if (p == 1 or prob == 1) {
        return fsize;
    }
    var q: f64 = 0;
    var sum: f64 = 0;
    while (true) : (q += 1) {
        sum += density.binomial(q, size, prob);
        if (sum >= p) {
            return q;
        }
    }
}

test "quantile.binomial" {
    try expectEqual(binomial(0  , 0 , 0.2), 0 );
    try expectEqual(binomial(0.5, 0 , 0.2), 0 );
    try expectEqual(binomial(1  , 0 , 0.2), 0 );
    try expectEqual(binomial(0  , 10, 0  ), 0 );
    try expectEqual(binomial(0.5, 10, 0  ), 0 );
    try expectEqual(binomial(1  , 10, 0  ), 10);
    try expectEqual(binomial(0  , 10, 1  ), 0 );
    try expectEqual(binomial(0.5, 10, 1  ), 10);
    try expectEqual(binomial(1  , 10, 1  ), 10);

    try expectApproxEqRel(binomial(0           , 10, 0.2), 0 );
    try expectApproxEqRel(binomial(0.1073741823, 10, 0.2), 0 );
    try expectApproxEqRel(binomial(0.1073741824, 10, 0.2), 0 );
    try expectApproxEqRel(binomial(0.1073741825, 10, 0.2), 1 );
    try expectApproxEqRel(binomial(0.3758096383, 10, 0.2), 1 );
    try expectApproxEqRel(binomial(0.3758096384, 10, 0.2), 1 );
    try expectApproxEqRel(binomial(0.3758096385, 10, 0.2), 2 );
    try expectApproxEqRel(binomial(1           , 10, 0.2), 10);
}

/// Quantile function of Negative Binomial distribution
///
/// size ∈ {0,1,2,⋯}
///
/// prob ∈ (0,1]
pub fn negativeBinomial(p: f64, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(0 <= p and p <= 1);
    if (p == 0 or prob == 1 or size == 0) {
        return 0;
    }
    if (p == 1) {
        return inf;
    }
    var q: f64 = 0;
    var sum: f64 = 0;
    while (true) : (q += 1) {
        sum += density.negativeBinomial(q, size, prob);
        if (sum >= p) {
            return q;
        }
    }
}

test "quantile.negativeBinomial" {
    try expectEqual(negativeBinomial(0  , 0 , 0.2), 0);
    try expectEqual(negativeBinomial(0.5, 0 , 0.2), 0);
    try expectEqual(negativeBinomial(1  , 0 , 0.2), 0);
    try expectEqual(negativeBinomial(0  , 10, 1  ), 0);
    try expectEqual(negativeBinomial(0.5, 10, 1  ), 0);
    try expectEqual(negativeBinomial(1  , 10, 1  ), 0);

    try expectApproxEqRel(negativeBinomial(0           , 10, 0.2), 0  );
    try expectApproxEqRel(negativeBinomial(0.0000001023, 10, 0.2), 0  );
    try expectApproxEqRel(negativeBinomial(0.0000001024, 10, 0.2), 0  );
    try expectApproxEqRel(negativeBinomial(0.0000001025, 10, 0.2), 1  );
    try expectApproxEqRel(negativeBinomial(0.0000009215, 10, 0.2), 1  );
    try expectApproxEqRel(negativeBinomial(0.0000009216, 10, 0.2), 1  );
    try expectApproxEqRel(negativeBinomial(0.0000009217, 10, 0.2), 2  );
    try expectEqual      (negativeBinomial(1           , 10, 0.2), inf);
}

/// Quantile function of Exponential distribution
///
/// rate ∈ (0,∞)
pub fn exponential(p: f64, rate: f64) f64 {
    assert(isFinite(rate));
    assert(rate >= 0);
    assert(0 <= p and p <= 1);
    return -std.math.log1p(-p) / rate;
}

test "quantile.exponential" {
    try expectApproxEqRel(exponential(0  , 3), 0                  );
    try expectApproxEqRel(exponential(0.2, 3), 0.07438118377140325);
    try expectApproxEqRel(exponential(0.4, 3), 0.17027520792199691);
    try expectApproxEqRel(exponential(0.6, 3), 0.30543024395805174);
    try expectApproxEqRel(exponential(0.8, 3), 0.53647930414470013);
    try expectEqual      (exponential(1  , 3), inf                );
}

/// Quantile function of Weibull distribution
///
/// shape and rate ∈ (0,∞)
pub fn weibull(p: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(0 <= p and p <= 1);
    return std.math.pow(f64, -std.math.log1p(-p), 1 / shape) / rate;
}

test "quantile.weibull" {
    try expectApproxEqRel(weibull(0  , 3, 0.5), 0                );
    try expectApproxEqRel(weibull(0.2, 3, 0.5), 1.213085586248216);
    try expectApproxEqRel(weibull(0.4, 3, 0.5), 1.598775754926823);
    try expectApproxEqRel(weibull(0.6, 3, 0.5), 1.942559933595852);
    try expectApproxEqRel(weibull(0.8, 3, 0.5), 2.343804613759100);
    try expectEqual      (weibull(1  , 3, 0.5), inf              );
}

/// Quantile function of Cauchy distribution
///
/// location ∈ (-∞,∞)
///
/// scale ∈ (0,∞)
pub fn cauchy(p: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return -inf;
    }
    if (p == 1) {
        return inf;
    }
    return location + scale * @tan(std.math.pi * (p - 0.5));
}

test "quantile.cauchy" {
    try expectEqual      (cauchy(0  , 0, 1), -inf               );
    try expectApproxEqRel(cauchy(0.2, 0, 1), -1.3763819204711736);
    try expectApproxEqRel(cauchy(0.4, 0, 1), -0.3249196962329063);
    try expectApproxEqRel(cauchy(0.6, 0, 1),  0.3249196962329066);
    try expectApproxEqRel(cauchy(0.8, 0, 1),  1.3763819204711740);
    try expectEqual      (cauchy(1  , 0, 1),  inf               );
}

/// Quantile function of Logistic distribution
///
/// location ∈ (-∞,∞)
///
/// scale ∈ (0,∞)
pub fn logistic(p: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(0 <= p and p <= 1);
    return location + scale * @log(p / (1 - p));
}

test "quantile.logistic" {
    try expectEqual      (logistic(0  , 0, 1), -inf               );
    try expectApproxEqRel(logistic(0.2, 0, 1), -1.3862943611198906);
    try expectApproxEqRel(logistic(0.4, 0, 1), -0.4054651081081643);
    try expectApproxEqRel(logistic(0.6, 0, 1),  0.4054651081081648);
    try expectApproxEqRel(logistic(0.8, 0, 1),  1.3862943611198908);
    try expectEqual      (logistic(1  , 0, 1),  inf               );
}

/// Quantile function of Gamma distribution
///
/// shape and rate ∈ (0,∞)
pub fn gamma(p: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return 0;
    }
    if (p == 1) {
        return inf;
    }
    const q = stdprob.inverseComplementedIncompleteGamma(shape, 1 - p);
    return q / rate;
}

test "quantile.gamma" {
    try expectApproxEqRel(gamma(0  , 3, 5), 0                 );
    try expectApproxEqRel(gamma(0.2, 3, 5), 0.3070088405289287);
    try expectApproxEqRel(gamma(0.4, 3, 5), 0.4570153808006763);
    try expectApproxEqRel(gamma(0.6, 3, 5), 0.6210757194526701);
    try expectApproxEqRel(gamma(0.8, 3, 5), 0.8558059720250668);
    try expectEqual      (gamma(1  , 3, 5), inf               );
}

/// Quantile function of Chi Squared distribution
///
/// df ∈ (0,∞)
pub fn chiSquared(p: f64, df: f64) f64 {
    return gamma(p, 0.5 * df, 0.5);
}

test "quantile.chiSquared" {
    try expectApproxEqRel(chiSquared(0  , 3), 0                );
    try expectApproxEqRel(chiSquared(0.2, 3), 1.005174013052349);
    try expectApproxEqRel(chiSquared(0.4, 3), 1.869168403388716);
    try expectApproxEqRel(chiSquared(0.6, 3), 2.946166073101952);
    try expectApproxEqRel(chiSquared(0.8, 3), 4.641627676087445);
    try expectEqual      (chiSquared(1  , 3), inf              );
}

/// Quantile function of F distribution
///
/// df1 and df2 ∈ (0,∞)
pub fn F(p: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(0 <= p and p <= 1);
    const q = stdprob.inverseIncompleteBeta(0.5 * df2, 0.5 * df1, 1 - p);
    return df2 / df1 * (1 / q - 1);
}

test "quantile.F" {
    try expectApproxEqRel(F(0  , 3, 5), 0                 );
    try expectApproxEqRel(F(0.2, 3, 5), 0.3372475270245997);
    try expectApproxEqRel(F(0.4, 3, 5), 0.6821342707772098);
    try expectApproxEqRel(F(0.6, 3, 5), 1.1978047828924259);
    try expectApproxEqRel(F(0.8, 3, 5), 2.2530173716474851);
    try expectEqual      (F(1  , 3, 5), inf               );
}

/// Quantile function of Beta distribution
///
/// shape1 and shape2 ∈ (0,∞)
pub fn beta(p: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(0 <= p and p <= 1);
    return stdprob.inverseIncompleteBeta(shape1, shape2, p);
}

test "quantile.beta" {
    try expectApproxEqRel(beta(0  , 3, 5), 0                 );
    try expectApproxEqRel(beta(0.2, 3, 5), 0.2283264643498391);
    try expectApproxEqRel(beta(0.4, 3, 5), 0.3205858305642004);
    try expectApproxEqRel(beta(0.6, 3, 5), 0.4092151219095550);
    try expectApproxEqRel(beta(0.8, 3, 5), 0.5167577700975785);
    try expectApproxEqRel(beta(1  , 3, 5), 1                 );
}

/// Quantile function of Normal distribution
///
/// mean ∈ (-∞,∞)
///
/// sd ∈ (0,∞)
pub fn normal(p: f64, mean: f64, sd: f64) f64 {
    assert(isFinite(mean) and isFinite(sd));
    assert(sd > 0);
    assert(0 <= p and p <= 1);
    return mean + sd * stdprob.inverseNormalDist(p);
}

test "quantile.normal" {
    try expectEqual      (normal(0  , 0, 1), -inf               );
    try expectApproxEqRel(normal(0.2, 0, 1), -0.8416212335729142);
    try expectApproxEqRel(normal(0.4, 0, 1), -0.2533471031357998);
    try expectApproxEqRel(normal(0.6, 0, 1),  0.2533471031358001);
    try expectApproxEqRel(normal(0.8, 0, 1),  0.8416212335729144);
    try expectEqual      (normal(1  , 0, 1),  inf               );
}

/// Quantile function of Log-normal distribution
///
/// meanlog ∈ (-∞,∞)
///
/// sdlog ∈ (0,∞)
pub fn logNormal(p: f64, meanlog: f64, sdlog: f64) f64 {
    assert(isFinite(meanlog) and isFinite(sdlog));
    assert(sdlog > 0);
    assert(0 <= p and p <= 1);
    const q = stdprob.inverseNormalDist(p);
    return @exp(meanlog + sdlog * q);
}

test "quantile.logNormal" {
    try expectApproxEqRel(logNormal(0  , 0, 1), 0                 );
    try expectApproxEqRel(logNormal(0.2, 0, 1), 0.4310111868818386);
    try expectApproxEqRel(logNormal(0.4, 0, 1), 0.7761984141563506);
    try expectApproxEqRel(logNormal(0.6, 0, 1), 1.2883303827500079);
    try expectApproxEqRel(logNormal(0.8, 0, 1), 2.3201253945043181);
    try expectEqual      (logNormal(1  , 0, 1), inf               );
}

/// Quantile function of t distribution
///
/// df ∈ (0,∞)
pub fn t(p: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(0 <= p and p <= 1);
    if (p < 0.5) {
        const q = stdprob.inverseIncompleteBeta(0.5 * df, 0.5, 2 * p);
        return -@sqrt(df / q - df);
    } else {
        const q = stdprob.inverseIncompleteBeta(0.5 * df, 0.5, 2 - 2 * p);
        return @sqrt(df / q - df);
    }
}

test "quantile.t" {
    try expectEqual      (t(0  , 3), -inf               );
    try expectApproxEqRel(t(0.2, 3), -0.9784723123633045);
    try expectApproxEqRel(t(0.4, 3), -0.2766706623326898);
    try expectApproxEqRel(t(0.6, 3),  0.2766706623326902);
    try expectApproxEqRel(t(0.8, 3),  0.9784723123633039);
    try expectEqual      (t(1  , 3),  inf               );
}
