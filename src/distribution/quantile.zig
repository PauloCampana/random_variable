//! Quantile functions Q(p).
//!
//! Represents the inverse of the probability function:
//! if Q(p) = s, then F(s) = p.
//!
//! Maps probabilities in [0,1] to numbers in the distribution's support.

const std = @import("std");
const stdprob = @import("../thirdyparty/prob.zig");

const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

/// prob ∈ [0,1]
pub fn bernoulli(p: f64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(0 <= p and p <= 1);
    return if (p > 1 - prob) 1 else 0;
}

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

/// lambda ∈ (0,∞)
pub fn poisson(p: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(0 <= p and p <= 1);
    if (p == 1) {
        return inf;
    }
    var mass = @exp(-lambda);
    var cumu = mass;
    var poi: f64 = 1;
    while (p >= cumu) : (poi += 1) {
        mass *= lambda / poi;
        cumu += mass;
    }
    return poi - 1;
}

/// size ∈ {0,1,2,⋯}, prob ∈ [0,1]
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
    const n = @as(f64, @floatFromInt(size));
    const np1 = n + 1;
    const qrob = 1 - prob;
    const pq = prob / qrob;
    var mass = std.math.pow(f64, qrob, n);
    var cumu = mass;
    var bin: f64 = 1;
    while (p >= cumu) : (bin += 1) {
        mass *= pq * (np1 - bin) / bin;
        cumu += mass;
    }
    return bin - 1;
}

/// size ∈ {1,2,3,⋯}, prob ∈ (0,1]
pub fn negativeBinomial(p: f64, size: u64, prob: f64) f64 {
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
    const nm1 = n - 1;
    const qrob = 1 - prob;
    var mass = std.math.pow(f64, prob, n);
    var cumu = mass;
    var nbi: f64 = 1;
    while (p >= cumu) : (nbi += 1) {
        mass *= qrob * (nm1 + nbi) / nbi;
        cumu += mass;
    }
    return nbi - 1;
}

/// min and max ∈ (-∞,∞)
pub fn uniform(p: f64, min: f64, max: f64) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(0 <= p and p <= 1);
    return min + (max - min) * p;
}

/// rate ∈ (0,∞)
pub fn exponential(p: f64, rate: f64) f64 {
    assert(isFinite(rate));
    assert(rate >= 0);
    assert(0 <= p and p <= 1);
    const q = -std.math.log1p(-p);
    return q / rate;
}

/// shape and rate ∈ (0,∞)
pub fn weibull(p: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(0 <= p and p <= 1);
    const q1 = -std.math.log1p(-p);
    const q2 = std.math.pow(f64, q1, 1 / shape);
    return q2 / rate;
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
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
    const q = @tan(std.math.pi * (p - 0.5));
    return location + scale * q;
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn logistic(p: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(0 <= p and p <= 1);
    const q = @log(p / (1 - p));
    return location + scale * q;
}

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

/// df ∈ (0,∞)
pub fn chiSquared(p: f64, df: f64) f64 {
    return gamma(p, 0.5 * df, 0.5);
}

/// df1 and df2 ∈ (0,∞)
pub fn f(p: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(0 <= p and p <= 1);
    const q = stdprob.inverseIncompleteBeta(0.5 * df2, 0.5 * df1, 1 - p);
    return (df2 / q - df2) / df1;
}

/// shape1 and shape2 ∈ (0,∞)
pub fn beta(p: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(0 <= p and p <= 1);
    return stdprob.inverseIncompleteBeta(shape1, shape2, p);
}

/// mean ∈ (-∞,∞), sd ∈ (0,∞)
pub fn normal(p: f64, mean: f64, sd: f64) f64 {
    assert(isFinite(mean) and isFinite(sd));
    assert(sd > 0);
    assert(0 <= p and p <= 1);
    const q = stdprob.inverseNormalDist(p);
    return mean + sd * q;
}

/// meanlog ∈ (-∞,∞), sdlog ∈ (0,∞)
pub fn logNormal(p: f64, meanlog: f64, sdlog: f64) f64 {
    assert(isFinite(meanlog) and isFinite(sdlog));
    assert(sdlog > 0);
    assert(0 <= p and p <= 1);
    const q = stdprob.inverseNormalDist(p);
    return @exp(meanlog + sdlog * q);
}

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

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "quantile.bernoulli" {
    try expectApproxEqRel(@as(f64, 0), bernoulli(0   , 0.2), eps);
    try expectApproxEqRel(@as(f64, 0), bernoulli(0.79, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0), bernoulli(0.8 , 0.2), eps);
    try expectApproxEqRel(@as(f64, 1), bernoulli(0.81, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1), bernoulli(1   , 0.2), eps);
}

test "quantile.geometric" {
    try expectApproxEqRel(@as(f64, 0  ), geometric(0   , 0.2), eps);
    try expectApproxEqRel(@as(f64, 0  ), geometric(0.19, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0  ), geometric(0.2 , 0.2), eps);
    try expectApproxEqRel(@as(f64, 1  ), geometric(0.21, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1  ), geometric(0.35, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1  ), geometric(0.36, 0.2), eps);
    try expectApproxEqRel(@as(f64, 2  ), geometric(0.37, 0.2), eps);
    try expectEqual      (@as(f64, inf), geometric(1   , 0.2)     );
}

test "quantile.poisson" {
    try expectApproxEqRel(@as(f64, 0  ), poisson(0                 , 3), eps);
    try expectApproxEqRel(@as(f64, 0  ), poisson(0.0497870683678638, 3), eps);
    try expectApproxEqRel(@as(f64, 0  ), poisson(0.0497870683678639, 3), eps);
    try expectApproxEqRel(@as(f64, 1  ), poisson(0.0497870683678640, 3), eps);
    try expectApproxEqRel(@as(f64, 1  ), poisson(0.1991482734714556, 3), eps);
    try expectApproxEqRel(@as(f64, 1  ), poisson(0.1991482734714557, 3), eps);
    try expectApproxEqRel(@as(f64, 2  ), poisson(0.1991482734714558, 3), eps);
    try expectEqual      (@as(f64, inf), poisson(1                 , 3)     );
}

test "quantile.binomial" {
    try expectEqual(@as(f64, 0 ), binomial(0  , 0 , 0.2));
    try expectEqual(@as(f64, 0 ), binomial(0.5, 0 , 0.2));
    try expectEqual(@as(f64, 0 ), binomial(1  , 0 , 0.2));
    try expectEqual(@as(f64, 0 ), binomial(0  , 10, 0  ));
    try expectEqual(@as(f64, 0 ), binomial(0.5, 10, 0  ));
    try expectEqual(@as(f64, 10), binomial(1  , 10, 0  ));
    try expectEqual(@as(f64, 0 ), binomial(0  , 10, 1  ));
    try expectEqual(@as(f64, 10), binomial(0.5, 10, 1  ));
    try expectEqual(@as(f64, 10), binomial(1  , 10, 1  ));

    try expectApproxEqRel(@as(f64, 0 ), binomial(0           , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0 ), binomial(0.1073741823, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0 ), binomial(0.1073741824, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1 ), binomial(0.1073741825, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1 ), binomial(0.3758096383, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1 ), binomial(0.3758096384, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 2 ), binomial(0.3758096385, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 10), binomial(1           , 10, 0.2), eps);
}

test "quantile.negativeBinomial" {
    try expectEqual(@as(f64, 0), negativeBinomial(0  , 10, 1));
    try expectEqual(@as(f64, 0), negativeBinomial(0.5, 10, 1));
    try expectEqual(@as(f64, 0), negativeBinomial(1  , 10, 1));

    try expectApproxEqRel(@as(f64, 0  ), negativeBinomial(0           , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0  ), negativeBinomial(0.0000001023, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0  ), negativeBinomial(0.0000001024, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1  ), negativeBinomial(0.0000001025, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1  ), negativeBinomial(0.0000009215, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 1  ), negativeBinomial(0.0000009216, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 2  ), negativeBinomial(0.0000009217, 10, 0.2), eps);
    try expectEqual      (@as(f64, inf), negativeBinomial(1           , 10, 0.2)     );
}

test "quantile.uniform" {
    try expectApproxEqRel(@as(f64, 3  ), uniform(0  , 3, 5), eps);
    try expectApproxEqRel(@as(f64, 3.4), uniform(0.2, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 3.8), uniform(0.4, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 4.2), uniform(0.6, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 4.6), uniform(0.8, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 5  ), uniform(1  , 3, 5), eps);
}

test "quantile.exponential" {
    try expectApproxEqRel(@as(f64, 0                  ), exponential(0  , 3), eps);
    try expectApproxEqRel(@as(f64, 0.07438118377140325), exponential(0.2, 3), eps);
    try expectApproxEqRel(@as(f64, 0.17027520792199691), exponential(0.4, 3), eps);
    try expectApproxEqRel(@as(f64, 0.30543024395805174), exponential(0.6, 3), eps);
    try expectApproxEqRel(@as(f64, 0.53647930414470013), exponential(0.8, 3), eps);
    try expectEqual      (@as(f64, inf                ), exponential(1  , 3)     );
}

test "quantile.weibull" {
    try expectApproxEqRel(@as(f64, 0                ), weibull(0  , 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 1.213085586248216), weibull(0.2, 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 1.598775754926823), weibull(0.4, 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 1.942559933595852), weibull(0.6, 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 2.343804613759100), weibull(0.8, 3, 0.5), eps);
    try expectEqual      (@as(f64, inf              ), weibull(1  , 3, 0.5)     );
}

test "quantile.cauchy" {
    try expectEqual      (@as(f64, -inf               ), cauchy(0  , 0, 1)     );
    try expectApproxEqRel(@as(f64, -1.3763819204711736), cauchy(0.2, 0, 1), eps);
    try expectApproxEqRel(@as(f64, -0.3249196962329063), cauchy(0.4, 0, 1), eps);
    try expectApproxEqRel(@as(f64,  0.3249196962329066), cauchy(0.6, 0, 1), eps);
    try expectApproxEqRel(@as(f64,  1.3763819204711740), cauchy(0.8, 0, 1), eps);
    try expectEqual      (@as(f64,  inf               ), cauchy(1  , 0, 1)     );
}

test "quantile.logistic" {
    try expectEqual      (@as(f64, -inf               ), logistic(0  , 0, 1)     );
    try expectApproxEqRel(@as(f64, -1.3862943611198906), logistic(0.2, 0, 1), eps);
    try expectApproxEqRel(@as(f64, -0.4054651081081643), logistic(0.4, 0, 1), eps);
    try expectApproxEqRel(@as(f64,  0.4054651081081648), logistic(0.6, 0, 1), eps);
    try expectApproxEqRel(@as(f64,  1.3862943611198908), logistic(0.8, 0, 1), eps);
    try expectEqual      (@as(f64,  inf               ), logistic(1  , 0, 1)     );
}

test "quantile.gamma" {
    try expectApproxEqRel(@as(f64, 0                 ), gamma(0  , 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.3070088405289287), gamma(0.2, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.4570153808006763), gamma(0.4, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.6210757194526701), gamma(0.6, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.8558059720250668), gamma(0.8, 3, 5), eps);
    try expectEqual      (@as(f64, inf               ), gamma(1  , 3, 5)     );
}

test "quantile.chiSquared" {
    try expectApproxEqRel(@as(f64, 0                ), chiSquared(0  , 3), eps);
    try expectApproxEqRel(@as(f64, 1.005174013052349), chiSquared(0.2, 3), eps);
    try expectApproxEqRel(@as(f64, 1.869168403388716), chiSquared(0.4, 3), eps);
    try expectApproxEqRel(@as(f64, 2.946166073101952), chiSquared(0.6, 3), eps);
    try expectApproxEqRel(@as(f64, 4.641627676087445), chiSquared(0.8, 3), eps);
    try expectEqual      (@as(f64, inf              ), chiSquared(1  , 3)     );
}

test "quantile.f" {
    try expectApproxEqRel(@as(f64, 0                 ), f(0  , 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.3372475270245997), f(0.2, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.6821342707772098), f(0.4, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 1.1978047828924259), f(0.6, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 2.2530173716474851), f(0.8, 3, 5), eps);
    try expectEqual      (@as(f64, inf               ), f(1  , 3, 5)     );
}

test "quantile.beta" {
    try expectApproxEqRel(@as(f64, 0                 ), beta(0  , 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.2283264643498391), beta(0.2, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.3205858305642004), beta(0.4, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.4092151219095550), beta(0.6, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.5167577700975785), beta(0.8, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 1                 ), beta(1  , 3, 5), eps);
}

test "quantile.normal" {
    try expectEqual      (@as(f64, -inf               ), normal(0  , 0, 1)     );
    try expectApproxEqRel(@as(f64, -0.8416212335729142), normal(0.2, 0, 1), eps);
    try expectApproxEqRel(@as(f64, -0.2533471031357998), normal(0.4, 0, 1), eps);
    try expectApproxEqRel(@as(f64,  0.2533471031358001), normal(0.6, 0, 1), eps);
    try expectApproxEqRel(@as(f64,  0.8416212335729144), normal(0.8, 0, 1), eps);
    try expectEqual      (@as(f64,  inf               ), normal(1  , 0, 1)     );
}

test "quantile.logNormal" {
    try expectApproxEqRel(@as(f64, 0                 ), logNormal(0  , 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.4310111868818386), logNormal(0.2, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.7761984141563506), logNormal(0.4, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 1.2883303827500079), logNormal(0.6, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 2.3201253945043181), logNormal(0.8, 0, 1), eps);
    try expectEqual      (@as(f64, inf               ), logNormal(1  , 0, 1)     );
}

test "quantile.t" {
    try expectEqual      (@as(f64, -inf               ), t(0  , 3)     );
    try expectApproxEqRel(@as(f64, -0.9784723123633045), t(0.2, 3), eps);
    try expectApproxEqRel(@as(f64, -0.2766706623326898), t(0.4, 3), eps);
    try expectApproxEqRel(@as(f64,  0.2766706623326902), t(0.6, 3), eps);
    try expectApproxEqRel(@as(f64,  0.9784723123633039), t(0.8, 3), eps);
    try expectEqual      (@as(f64,  inf               ), t(1  , 3)     );
}
