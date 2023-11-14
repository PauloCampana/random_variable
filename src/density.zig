//! Probability density/mass functions f(x)/P(x)
//! for common probability distributions
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe
//! such as ±NaN, ±Inf, probabilities outside [0,1],
//! negative or zero shape, df, rate or scale parameters.

const std = @import("std");
const lnGamma = @import("thirdyparty/prob.zig").lnGamma;
const expectEqual = @import("thirdyparty/testing.zig").expectEqual;
const expectApproxEqRel = @import("thirdyparty/testing.zig").expectApproxEqRel;

const assert = std.debug.assert;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const isFinite = std.math.isFinite; // tests false for both inf and nan
const inf = std.math.inf(f64);

/// Probability density function of Uniform distribution
///
/// min and max ∈ (-∞,∞)
pub fn uniform(x: f64, min: f64, max: f64) f64 {
    assert(isFinite(min) and isFinite(max));
    assert(!isNan(x));
    if (x < min or x > max) {
        return 0;
    }
    return 1 / (max - min);
}

test "density.uniform" {
    try expectEqual(uniform(-inf, 0, 1), 0);
    try expectEqual(uniform( inf, 0, 1), 0);

    try expectApproxEqRel(uniform(2, 3, 5), 0  );
    try expectApproxEqRel(uniform(3, 3, 5), 0.5);
    try expectApproxEqRel(uniform(4, 3, 5), 0.5);
    try expectApproxEqRel(uniform(5, 3, 5), 0.5);
    try expectApproxEqRel(uniform(6, 3, 5), 0  );
}

/// Probability mass function of Bernoulli distribution
///
/// prob ∈ [0,1]
pub fn bernoulli(x: f64, prob: f64) f64 {
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
    try expectEqual(bernoulli(-inf, 0.2), 0);
    try expectEqual(bernoulli( inf, 0.2), 0);

    try expectApproxEqRel(bernoulli(-0.1, 0.2), 0  );
    try expectApproxEqRel(bernoulli( 0  , 0.2), 0.8);
    try expectApproxEqRel(bernoulli( 0.1, 0.2), 0  );
    try expectApproxEqRel(bernoulli( 0.9, 0.2), 0  );
    try expectApproxEqRel(bernoulli( 1  , 0.2), 0.2);
    try expectApproxEqRel(bernoulli( 1.1, 0.2), 0  );
}

/// Probability mass function of Geometric distribution
///
/// prob ∈ (0,1]
pub fn geometric(x: f64, prob: f64) f64 {
    assert(isFinite(prob));
    assert(0 < prob and prob <= 1);
    assert(!isNan(x));
    if (x < 0 or x != @round(x)) {
        return 0;
    }
    return prob * std.math.pow(f64, (1 - prob), x);
}

test "density.geometric" {
    try expectEqual(geometric(-inf, 0.2), 0);
    try expectEqual(geometric( inf, 0.2), 0);
    try expectEqual(geometric(-inf, 1  ), 0);
    try expectEqual(geometric( inf, 1  ), 0);

    try expectEqual(geometric(0, 1), 1);
    try expectEqual(geometric(1, 1), 0);

    try expectApproxEqRel(geometric(-0.1, 0.2), 0   );
    try expectApproxEqRel(geometric( 0  , 0.2), 0.2 );
    try expectApproxEqRel(geometric( 0.1, 0.2), 0   );
    try expectApproxEqRel(geometric( 0.9, 0.2), 0   );
    try expectApproxEqRel(geometric( 1  , 0.2), 0.16);
    try expectApproxEqRel(geometric( 1.1, 0.2), 0   );
}

/// Probability mass function of Poisson distribution
///
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

test "density.poisson" {
    try expectEqual(poisson(-inf, 3), 0);
    try expectEqual(poisson( inf, 3), 0);

    try expectApproxEqRel(poisson(-0.1, 3), 0                 );
    try expectApproxEqRel(poisson( 0  , 3), 0.0497870683678639);
    try expectApproxEqRel(poisson( 0.1, 3), 0                 );
    try expectApproxEqRel(poisson( 0.9, 3), 0                 );
    try expectApproxEqRel(poisson( 1  , 3), 0.1493612051035919);
    try expectApproxEqRel(poisson( 1.1, 3), 0                 );
}

/// Probability mass function of Binomial distribution
///
/// size ∈ {0,1,2,⋯}
///
/// prob ∈ [0,1]
pub fn binomial(x: f64, size: u64, prob: f64) f64 {
    assert(isFinite(prob));
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
    const log = binom + x * @log(prob) + diff * @log(1 - prob);
    return @exp(log);
}

test "density.binomial" {
    try expectEqual(binomial(-inf, 10, 0.2), 0);
    try expectEqual(binomial( inf, 10, 0.2), 0);

    try expectEqual(binomial( 0,  0, 0.2), 1);
    try expectEqual(binomial( 1,  0, 0.2), 0);
    try expectEqual(binomial( 0, 10, 0  ), 1);
    try expectEqual(binomial( 1, 10, 0  ), 0);
    try expectEqual(binomial( 9, 10, 1  ), 0);
    try expectEqual(binomial(10, 10, 1  ), 1);
    try expectEqual(binomial(11, 10, 1  ), 0);

    try expectApproxEqRel(binomial(-0.1, 10, 0.2), 0           );
    try expectApproxEqRel(binomial( 0  , 10, 0.2), 0.1073741824);
    try expectApproxEqRel(binomial( 0.1, 10, 0.2), 0           );
    try expectApproxEqRel(binomial( 0.9, 10, 0.2), 0           );
    try expectApproxEqRel(binomial( 1  , 10, 0.2), 0.2684354560);
    try expectApproxEqRel(binomial( 1.1, 10, 0.2), 0           );
}

/// Probability mass function of Negative Binomial distribution
///
/// size ∈ {0,1,2,⋯}
///
/// prob ∈ (0,1]
pub fn negativeBinomial(x: f64, size: u64, prob: f64) f64 {
    assert(isFinite(prob));
    assert(0 < prob and prob <= 1);
    assert(!isNan(x));
    if (x < 0 or isInf(x) or x != @round(x)) {
        return 0;
    }
    if (prob == 1 or size == 0) {
        return if (x == 0) 1 else 0;
    }
    const fsize = @as(f64, @floatFromInt(size));
    const binom = lnGamma(fsize + x) - lnGamma(fsize) - lnGamma(x + 1);
    const log = binom + fsize * @log(prob) + x * @log(1 - prob);
    return @exp(log);
}

test "density.negativeBinomial" {
    try expectEqual(negativeBinomial(-inf, 10, 0.2), 0);
    try expectEqual(negativeBinomial( inf, 10, 0.2), 0);

    try expectEqual(negativeBinomial( 0,  0, 0.2), 1);
    try expectEqual(negativeBinomial( 1,  0, 0.2), 0);
    try expectEqual(negativeBinomial( 0, 10, 1  ), 1);
    try expectEqual(negativeBinomial( 1, 10, 1  ), 0);

    try expectApproxEqRel(negativeBinomial(-0.1, 10, 0.2), 0           );
    try expectApproxEqRel(negativeBinomial( 0  , 10, 0.2), 0.0000001024);
    try expectApproxEqRel(negativeBinomial( 0.1, 10, 0.2), 0           );
    try expectApproxEqRel(negativeBinomial( 0.9, 10, 0.2), 0           );
    try expectApproxEqRel(negativeBinomial( 1  , 10, 0.2), 0.0000008192);
    try expectApproxEqRel(negativeBinomial( 1.1, 10, 0.2), 0           );
}

/// Probability density function of Exponential distribution
///
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

test "density.exponential" {
    try expectEqual(exponential(-inf, 3), 0);
    try expectEqual(exponential( inf, 3), 0);

    try expectApproxEqRel(exponential(0, 3), 3                   );
    try expectApproxEqRel(exponential(1, 3), 0.149361205103591900);
    try expectApproxEqRel(exponential(2, 3), 0.007436256529999075);
}

/// Probability density function of Weibull distribution
///
/// shape and rate ∈ (0,∞)
pub fn weibull(x: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(x));
    if (x == 0) {
        return if (shape < 1) inf else 0;
    }
    if (x < 0 or isInf(x)) {
        return 0;
    }
    const z = rate * x;
    const zam1 = std.math.pow(f64, z, shape - 1);
    const za = zam1 * z;
    return shape * rate * @exp(-za) * zam1;
}

test "density.weibull" {
    try expectEqual(weibull(-inf, 3, 0.5), 0);
    try expectEqual(weibull( inf, 3, 0.5), 0);

    try expectApproxEqRel(weibull(0, 3, 0.5), 0                 );
    try expectApproxEqRel(weibull(1, 3, 0.5), 0.3309363384692233);
    try expectApproxEqRel(weibull(2, 3, 0.5), 0.5518191617571635);
}

/// Probability density function of Cauchy distribution
///
/// location ∈ (-∞,∞)
///
/// scale ∈ (0,∞)
pub fn cauchy(x: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    const z = (x - location) / scale;
    return 1 / (std.math.pi * scale * (1 + z * z));
}

test "density.cauchy" {
    try expectEqual(cauchy(-inf, 0, 1), 0);
    try expectEqual(cauchy( inf, 0, 1), 0);

    try expectApproxEqRel(cauchy(0, 0, 1), 0.3183098861837906);
    try expectApproxEqRel(cauchy(1, 0, 1), 0.1591549430918953);
    try expectApproxEqRel(cauchy(2, 0, 1), 0.0636619772367581);
}

/// Probability density function of Logistic distribution
///
/// location ∈ (-∞,∞)
///
/// scale ∈ (0,∞)
pub fn logistic(x: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    const z = @abs(x - location) / scale;
    const expz = @exp(-z);
    const expzp1 = expz + 1;
    return expz / (scale * expzp1 * expzp1);
}

test "density.logistic" {
    try expectEqual(logistic(-inf, 0, 1), 0);
    try expectEqual(logistic( inf, 0, 1), 0);

    try expectApproxEqRel(logistic(0, 0, 1), 0.25              );
    try expectApproxEqRel(logistic(1, 0, 1), 0.1966119332414819);
    try expectApproxEqRel(logistic(2, 0, 1), 0.1049935854035065);
}

/// Probability density function of Gamma distribution
///
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

test "density.gamma" {
    try expectEqual(gamma(-inf, 3, 5), 0);
    try expectEqual(gamma( inf, 3, 5), 0);

    try expectEqual(gamma(0, 0.9, 5), inf);
    try expectEqual(gamma(0, 1  , 5), 5  );
    try expectEqual(gamma(0, 1.1, 5), 0  );

    try expectApproxEqRel(gamma(0, 3, 5), 0                 );
    try expectApproxEqRel(gamma(1, 3, 5), 0.4211216874428417);
    try expectApproxEqRel(gamma(2, 3, 5), 0.0113499824406212);
}

/// Probability density function of Chi Squared distribution
///
/// df ∈ (0,∞)
pub fn chiSquared(x: f64, df: f64) f64 {
    return gamma(x, 0.5 * df, 0.5);
}

test "density.chiSquared" {
    try expectEqual(chiSquared(-inf, 3), 0);
    try expectEqual(chiSquared( inf, 3), 0);

    try expectEqual(chiSquared(0, 1.8), inf);
    try expectEqual(chiSquared(0, 2  ), 0.5);
    try expectEqual(chiSquared(0, 2.2), 0  );

    try expectApproxEqRel(chiSquared(0, 3), 0                 );
    try expectApproxEqRel(chiSquared(1, 3), 0.2419707245191434);
    try expectApproxEqRel(chiSquared(2, 3), 0.2075537487102973);
}

/// Probability density function of F distribution
///
/// df1 and df2 ∈ (0,∞)
pub fn F(x: f64, df1: f64, df2: f64) f64 {
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

test "density.F" {
    try expectEqual(F(-inf, 3, 5), 0);
    try expectEqual(F( inf, 3, 5), 0);

    try expectEqual(F(0, 1.8, 5), inf);
    try expectEqual(F(0, 2  , 5), 1  );
    try expectEqual(F(0, 2.2, 5), 0  );

    try expectApproxEqRel(F(0, 3, 5), 0                 );
    try expectApproxEqRel(F(1, 3, 5), 0.3611744789422851);
    try expectApproxEqRel(F(2, 3, 5), 0.1428963909075316);
}

/// Probability density function of Beta distribution
///
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

test "density.beta" {
    try expectEqual(beta(-inf, 3, 5), 0);
    try expectEqual(beta( inf, 3, 5), 0);

    try expectEqual(beta(0, 0.9, 5  ), inf);
    try expectEqual(beta(0, 1  , 5  ), 5  );
    try expectEqual(beta(0, 1.1, 5  ), 0  );
    try expectEqual(beta(1, 3  , 0.9), inf);
    try expectEqual(beta(1, 3  , 1  ), 3  );
    try expectEqual(beta(1, 3  , 1.1), 0  );

    try expectApproxEqRel(beta(0  , 3, 5), 0      );
    try expectApproxEqRel(beta(0.2, 3, 5), 1.72032);
    try expectApproxEqRel(beta(0.8, 3, 5), 0.10752);
    try expectApproxEqRel(beta(1  , 3, 5), 0      );
}

/// Probability density function of Normal distribution
///
/// mean ∈ (-∞,∞)
///
/// sd ∈ (0,∞)
pub fn normal(x: f64, mean: f64, sd: f64) f64 {
    assert(isFinite(mean) and isFinite(sd));
    assert(sd > 0);
    assert(!isNan(x));
    const z = (x - mean) / sd;
    const sqrt2pi = comptime @sqrt(2 * std.math.pi);
    return @exp(-0.5 * z * z) / (sd * sqrt2pi);
}

test "density.normal" {
    try expectEqual(normal(-inf, 0, 1), 0);
    try expectEqual(normal( inf, 0, 1), 0);

    try expectApproxEqRel(normal(0, 0, 1), 0.3989422804014327);
    try expectApproxEqRel(normal(1, 0, 1), 0.2419707245191433);
    try expectApproxEqRel(normal(2, 0, 1), 0.0539909665131880);
}

/// Probability density function of Log-normal distribution
///
/// meanlog ∈ (-∞,∞)
///
/// sdlog ∈ (0,∞)
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

test "density.logNormal" {
    try expectEqual(logNormal(-inf, 0, 1), 0);
    try expectEqual(logNormal( inf, 0, 1), 0);

    try expectApproxEqRel(logNormal(0, 0, 1), 0                 );
    try expectApproxEqRel(logNormal(1, 0, 1), 0.3989422804014327);
    try expectApproxEqRel(logNormal(2, 0, 1), 0.1568740192789811);
}

/// Probability density function of t distribution
///
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

test "density.t" {
    try expectEqual(t(-inf, 3), 0);
    try expectEqual(t( inf, 3), 0);

    try expectApproxEqRel(t(0, 3), 0.3675525969478613);
    try expectApproxEqRel(t(1, 3), 0.2067483357831720);
    try expectApproxEqRel(t(2, 3), 0.0675096606638929);
}
