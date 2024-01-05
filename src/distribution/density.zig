//! Probability density functions f(x),
//! known as probability mass for discrete distributions.
//!
//! For a discrete random variable X, represents the probability
//! of X assuming a certain value: f(x) = P(X = x).
//!
//! Maps every number to a probability,
//! 0 if the number is not in the support of the distribution.

const std = @import("std");
const lnGamma = @import("../thirdyparty/prob.zig").lnGamma;

const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

/// prob ∈ [0,1]
pub fn bernoulli(x: f64, prob: f64) f64 {
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

/// prob ∈ (0,1]
pub fn geometric(x: f64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(!isNan(x));
    if (x < 0 or x != @round(x)) {
        return 0;
    }
    return prob * std.math.pow(f64, (1 - prob), x);
}

/// lambda ∈ (0,∞)
pub fn poisson(x: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x) or x != @round(x)) {
        return 0;
    }
    const log = -lambda + x * @log(lambda) - lnGamma(x + 1);
    return @exp(log);
}

/// size ∈ {0,1,2,⋯}, prob ∈ [0,1]
pub fn binomial(x: f64, size: u64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    assert(!isNan(x));
    const fsize = @as(f64, @floatFromInt(size));
    if (x < 0 or x > fsize or x != @round(x)) {
        return 0;
    }
    if (prob == 0) {
        return if (x == 0) 1 else 0;
    }
    if (prob == 1) {
        return if (x == fsize) 1 else 0;
    }
    const diff = fsize - x;
    const binom = lnGamma(fsize + 1) - lnGamma(x + 1) - lnGamma(diff + 1);
    const log = binom + x * @log(prob) + diff * std.math.log1p(-prob);
    return @exp(log);
}

/// size ∈ {1,2,3,⋯}, prob ∈ (0,1]
pub fn negativeBinomial(x: f64, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x) or x != @round(x)) {
        return 0;
    }
    if (prob == 1) {
        return if (x == 0) 1 else 0;
    }
    const fsize = @as(f64, @floatFromInt(size));
    const binom = lnGamma(fsize + x) - lnGamma(fsize) - lnGamma(x + 1);
    const log = binom + fsize * @log(prob) + x * std.math.log1p(-prob);
    return @exp(log);
}

/// min and max ∈ (-∞,∞)
pub fn uniform(x: f64, min: f64, max: f64) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(!isNan(x));
    if (x < min or x > max) {
        return 0;
    }
    return 1 / (max - min);
}

/// rate ∈ (0,∞)
pub fn exponential(x: f64, rate: f64) f64 {
    assert(isFinite(rate));
    assert(rate > 0);
    assert(!isNan(x));
    if (x < 0) {
        return 0;
    }
    return rate * @exp(-rate * x);
}

/// shape and rate ∈ (0,∞)
pub fn weibull(x: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x)) {
        return 0;
    }
    if (x == 0) {
        if (shape == 1) {
            return rate;
        }
        return if (shape < 1) inf else 0;
    }
    const z = rate * x;
    const zam1 = std.math.pow(f64, z, shape - 1);
    const za = zam1 * z;
    return shape * rate * @exp(-za) * zam1;
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn cauchy(x: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    const z = (x - location) / scale;
    return 1 / (std.math.pi * scale * (1 + z * z));
}

/// location ∈ (-∞,∞), scale ∈ (0,∞)
pub fn logistic(x: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    const z = @abs(x - location) / scale;
    const expz = @exp(-z);
    const expzp1 = expz + 1;
    return expz / (scale * expzp1 * expzp1);
}

/// shape and rate ∈ (0,∞)
pub fn gamma(x: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x)) {
        return 0;
    }
    if (x == 0) {
        if (shape == 1) {
            return rate;
        }
        return if (shape < 1) inf else 0;
    }
    const z = rate * x;
    const den = lnGamma(shape) + @log(x);
    const num = shape * @log(z) - z;
    return @exp(num - den);
}

/// df ∈ (0,∞)
pub fn chiSquared(x: f64, df: f64) f64 {
    return gamma(x, 0.5 * df, 0.5);
}

/// df1 and df2 ∈ (0,∞)
pub fn f(x: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x)) {
        return 0;
    }
    if (x == 0) {
        if (df1 == 2) {
            return 1;
        }
        return if (df1 < 2) inf else 0;
    }
    const df3 = df1 / 2;
    const df4 = df2 / 2;
    const df5 = df3 + df4;
    const num1 = df3 * @log(df1) + df4 * @log(df2) + (df3 - 1) * @log(x);
    const num2 = -df5 * @log(df2 + df1 * x);
    const den = lnGamma(df3) + lnGamma(df4) - lnGamma(df5);
    return @exp(num1 + num2 - den);
}

/// shape1 and shape2 ∈ (0,∞)
pub fn beta(x: f64, shape1: f64, shape2: f64) f64 {
    assert(isFinite(shape1) and isFinite(shape2));
    assert(shape1 > 0 and shape2 > 0);
    assert(!isNan(x));
    if (x < 0 or x > 1) {
        return 0;
    }
    if (x == 0) {
        if (shape1 == 1) {
            return shape2;
        }
        return if (shape1 < 1) inf else 0;
    }
    if (x == 1) {
        if (shape2 == 1) {
            return shape1;
        }
        return if (shape2 < 1) inf else 0;
    }
    const num = (shape1 - 1) * @log(x) + (shape2 - 1) * std.math.log1p(-x);
    const den = lnGamma(shape1) + lnGamma(shape2) - lnGamma(shape1 + shape2);
    return @exp(num - den);
}

/// mean ∈ (-∞,∞), sd ∈ (0,∞)
pub fn normal(x: f64, mean: f64, sd: f64) f64 {
    assert(isFinite(mean) and isFinite(sd));
    assert(sd > 0);
    assert(!isNan(x));
    const z = (x - mean) / sd;
    const sqrt2pi = comptime @sqrt(2 * std.math.pi);
    return @exp(-0.5 * z * z) / (sd * sqrt2pi);
}

/// meanlog ∈ (-∞,∞), sdlog ∈ (0,∞)
pub fn logNormal(x: f64, meanlog: f64, sdlog: f64) f64 {
    assert(isFinite(meanlog) and isFinite(sdlog));
    assert(sdlog > 0);
    assert(!isNan(x));
    if (x <= 0) {
        return 0;
    }
    const z = (@log(x) - meanlog) / sdlog;
    const sqrt2pi = comptime @sqrt(2 * std.math.pi);
    return @exp(-0.5 * z * z) / (x * sdlog * sqrt2pi);
}

/// df ∈ (0,∞)
pub fn t(x: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(!isNan(x));
    const df2 = 0.5 * df + 0.5;
    const num = df2 * @log(df / (df + x * x)) - 0.5 * @log(df);
    const den = lnGamma(0.5 * df) + lnGamma(0.5) - lnGamma(df2);
    return @exp(num - den);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "density.bernoulli" {
    try expectEqual(@as(f64, 0), bernoulli(-inf, 0.2));
    try expectEqual(@as(f64, 0), bernoulli( inf, 0.2));

    try expectApproxEqRel(@as(f64, 0  ), bernoulli(-0.1, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.8), bernoulli( 0  , 0.2), eps);
    try expectApproxEqRel(@as(f64, 0  ), bernoulli( 0.1, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0  ), bernoulli( 0.9, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.2), bernoulli( 1  , 0.2), eps);
    try expectApproxEqRel(@as(f64, 0  ), bernoulli( 1.1, 0.2), eps);
}

test "density.geometric" {
    try expectEqual(@as(f64, 0), geometric(-inf, 0.2));
    try expectEqual(@as(f64, 0), geometric( inf, 0.2));
    try expectEqual(@as(f64, 0), geometric(-inf, 1  ));
    try expectEqual(@as(f64, 0), geometric( inf, 1  ));

    try expectEqual(@as(f64, 1), geometric(0, 1));
    try expectEqual(@as(f64, 0), geometric(1, 1));

    try expectApproxEqRel(@as(f64, 0   ), geometric(-0.1, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.2 ), geometric( 0  , 0.2), eps);
    try expectApproxEqRel(@as(f64, 0   ), geometric( 0.1, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0   ), geometric( 0.9, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.16), geometric( 1  , 0.2), eps);
    try expectApproxEqRel(@as(f64, 0   ), geometric( 1.1, 0.2), eps);
}

test "density.poisson" {
    try expectEqual(@as(f64, 0), poisson(-inf, 3));
    try expectEqual(@as(f64, 0), poisson( inf, 3));

    try expectApproxEqRel(@as(f64, 0                 ), poisson(-0.1, 3), eps);
    try expectApproxEqRel(@as(f64, 0.0497870683678639), poisson( 0  , 3), eps);
    try expectApproxEqRel(@as(f64, 0                 ), poisson( 0.1, 3), eps);
    try expectApproxEqRel(@as(f64, 0                 ), poisson( 0.9, 3), eps);
    try expectApproxEqRel(@as(f64, 0.1493612051035919), poisson( 1  , 3), eps);
    try expectApproxEqRel(@as(f64, 0                 ), poisson( 1.1, 3), eps);
}

test "density.binomial" {
    try expectEqual(@as(f64, 0), binomial(-inf, 10, 0.2));
    try expectEqual(@as(f64, 0), binomial( inf, 10, 0.2));

    try expectEqual(@as(f64, 1), binomial( 0,  0, 0.2));
    try expectEqual(@as(f64, 0), binomial( 1,  0, 0.2));
    try expectEqual(@as(f64, 1), binomial( 0, 10, 0  ));
    try expectEqual(@as(f64, 0), binomial( 1, 10, 0  ));
    try expectEqual(@as(f64, 0), binomial( 9, 10, 1  ));
    try expectEqual(@as(f64, 1), binomial(10, 10, 1  ));
    try expectEqual(@as(f64, 0), binomial(11, 10, 1  ));

    try expectApproxEqRel(@as(f64, 0           ), binomial(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.1073741824), binomial( 0  , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0           ), binomial( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0           ), binomial( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.2684354560), binomial( 1  , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0           ), binomial( 1.1, 10, 0.2), eps);
}

test "density.negativeBinomial" {
    try expectEqual(@as(f64, 0), negativeBinomial(-inf, 10, 0.2));
    try expectEqual(@as(f64, 0), negativeBinomial( inf, 10, 0.2));

    try expectEqual(@as(f64, 1), negativeBinomial( 0, 10, 1  ));
    try expectEqual(@as(f64, 0), negativeBinomial( 1, 10, 1  ));

    try expectApproxEqRel(@as(f64, 0           ), negativeBinomial(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.0000001024), negativeBinomial( 0  , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0           ), negativeBinomial( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0           ), negativeBinomial( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0.0000008192), negativeBinomial( 1  , 10, 0.2), eps);
    try expectApproxEqRel(@as(f64, 0           ), negativeBinomial( 1.1, 10, 0.2), eps);
}

test "density.uniform" {
    try expectEqual(@as(f64, 0), uniform(-inf, 0, 1));
    try expectEqual(@as(f64, 0), uniform( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0  ), uniform(2, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.5), uniform(3, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.5), uniform(4, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.5), uniform(5, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0  ), uniform(6, 3, 5), eps);
}

test "density.exponential" {
    try expectEqual(@as(f64, 0), exponential(-inf, 3));
    try expectEqual(@as(f64, 0), exponential( inf, 3));

    try expectApproxEqRel(@as(f64, 3                   ), exponential(0, 3), eps);
    try expectApproxEqRel(@as(f64, 0.149361205103591900), exponential(1, 3), eps);
    try expectApproxEqRel(@as(f64, 0.007436256529999075), exponential(2, 3), eps);
}

test "density.weibull" {
    try expectEqual(@as(f64, 0), weibull(-inf, 3, 0.5));
    try expectEqual(@as(f64, 0), weibull( inf, 3, 0.5));

    try expectEqual(@as(f64, inf), weibull(0, 0.9, 5));
    try expectEqual(@as(f64, 5  ), weibull(0, 1  , 5));
    try expectEqual(@as(f64, 0  ), weibull(0, 1.1, 5));

    try expectApproxEqRel(@as(f64, 0                 ), weibull(0, 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 0.3309363384692233), weibull(1, 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 0.5518191617571635), weibull(2, 3, 0.5), eps);
}

test "density.cauchy" {
    try expectEqual(@as(f64, 0), cauchy(-inf, 0, 1));
    try expectEqual(@as(f64, 0), cauchy( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0.3183098861837906), cauchy(0, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.1591549430918953), cauchy(1, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.0636619772367581), cauchy(2, 0, 1), eps);
}

test "density.logistic" {
    try expectEqual(@as(f64, 0), logistic(-inf, 0, 1));
    try expectEqual(@as(f64, 0), logistic( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0.25              ), logistic(0, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.1966119332414819), logistic(1, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.1049935854035065), logistic(2, 0, 1), eps);
}

test "density.gamma" {
    try expectEqual(@as(f64, 0), gamma(-inf, 3, 5));
    try expectEqual(@as(f64, 0), gamma( inf, 3, 5));

    try expectEqual(@as(f64, inf), gamma(0, 0.9, 5));
    try expectEqual(@as(f64, 5  ), gamma(0, 1  , 5));
    try expectEqual(@as(f64, 0  ), gamma(0, 1.1, 5));

    try expectApproxEqRel(@as(f64, 0                 ), gamma(0, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.4211216874428417), gamma(1, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.0113499824406212), gamma(2, 3, 5), eps);
}

test "density.chiSquared" {
    try expectEqual(@as(f64, 0), chiSquared(-inf, 3));
    try expectEqual(@as(f64, 0), chiSquared( inf, 3));

    try expectEqual(@as(f64, inf), chiSquared(0, 1.8));
    try expectEqual(@as(f64, 0.5), chiSquared(0, 2  ));
    try expectEqual(@as(f64, 0  ), chiSquared(0, 2.2));

    try expectApproxEqRel(@as(f64, 0                 ), chiSquared(0, 3), eps);
    try expectApproxEqRel(@as(f64, 0.2419707245191434), chiSquared(1, 3), eps);
    try expectApproxEqRel(@as(f64, 0.2075537487102973), chiSquared(2, 3), eps);
}

test "density.f" {
    try expectEqual(@as(f64, 0), f(-inf, 3, 5));
    try expectEqual(@as(f64, 0), f( inf, 3, 5));

    try expectEqual(@as(f64, inf), f(0, 1.8, 5));
    try expectEqual(@as(f64, 1  ), f(0, 2  , 5));
    try expectEqual(@as(f64, 0  ), f(0, 2.2, 5));

    try expectApproxEqRel(@as(f64, 0                 ), f(0, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.3611744789422851), f(1, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.1428963909075316), f(2, 3, 5), eps);
}

test "density.beta" {
    try expectEqual(@as(f64, 0), beta(-inf, 3, 5));
    try expectEqual(@as(f64, 0), beta( inf, 3, 5));

    try expectEqual(@as(f64, inf), beta(0, 0.9, 5  ));
    try expectEqual(@as(f64, 5  ), beta(0, 1  , 5  ));
    try expectEqual(@as(f64, 0  ), beta(0, 1.1, 5  ));
    try expectEqual(@as(f64, inf), beta(1, 3  , 0.9));
    try expectEqual(@as(f64, 3  ), beta(1, 3  , 1  ));
    try expectEqual(@as(f64, 0  ), beta(1, 3  , 1.1));

    try expectApproxEqRel(@as(f64, 0      ), beta(0  , 3, 5), eps);
    try expectApproxEqRel(@as(f64, 1.72032), beta(0.2, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0.10752), beta(0.8, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0      ), beta(1  , 3, 5), eps);
}

test "density.normal" {
    try expectEqual(@as(f64, 0), normal(-inf, 0, 1));
    try expectEqual(@as(f64, 0), normal( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0.3989422804014327), normal(0, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.2419707245191433), normal(1, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.0539909665131880), normal(2, 0, 1), eps);
}

test "density.logNormal" {
    try expectEqual(@as(f64, 0), logNormal(-inf, 0, 1));
    try expectEqual(@as(f64, 0), logNormal( inf, 0, 1));

    try expectApproxEqRel(@as(f64, 0                 ), logNormal(0, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.3989422804014327), logNormal(1, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0.1568740192789811), logNormal(2, 0, 1), eps);
}

test "density.t" {
    try expectEqual(@as(f64, 0), t(-inf, 3));
    try expectEqual(@as(f64, 0), t( inf, 3));

    try expectApproxEqRel(@as(f64, 0.3675525969478613), t(0, 3), eps);
    try expectApproxEqRel(@as(f64, 0.2067483357831720), t(1, 3), eps);
    try expectApproxEqRel(@as(f64, 0.0675096606638929), t(2, 3), eps);
}
