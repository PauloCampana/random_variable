//! Quantile functions Q(x)
//! for common probability distributions
//!
//! Q(x) = F⁻¹(x)
//!
//! x ∈ [0, 1]
//!
//! Q(x) ∈ [-∞, ∞]
//!
//! asserts invalid distribution parameters on Debug and ReleaseSafe

const std = @import("std");
const stdprob = @import("thirdyparty/prob.zig");
const density = @import("density.zig");

const assert = std.debug.assert;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const isFinite = std.math.isFinite; // tests false for both inf and nan

const testing = std.testing;
const epsilon = 3e-15;
const inf = std.math.inf(f64);

/// Quantile function of Uniform distribution
///
/// min and max ∈ (-∞, ∞)
pub fn uniform(p: f64, min: f64, max: f64) f64 {
    assert(isFinite(min));
    assert(isFinite(max));
    assert(0 <= p and p <= 1);
    return min + (max - min) * p;
}

test "quantile.uniform" {
    try testing.expectApproxEqRel(uniform(0  , 3, 5), 3  , epsilon);
    try testing.expectApproxEqRel(uniform(0.2, 3, 5), 3.4, epsilon);
    try testing.expectApproxEqRel(uniform(0.4, 3, 5), 3.8, epsilon);
    try testing.expectApproxEqRel(uniform(0.6, 3, 5), 4.2, epsilon);
    try testing.expectApproxEqRel(uniform(0.8, 3, 5), 4.6, epsilon);
    try testing.expectApproxEqRel(uniform(1  , 3, 5), 5  , epsilon);
}

/// Quantile function of Bernoulli distribution
///
/// prob ∈ [0, 1]
pub fn bernoulli(p: f64, prob: f64) f64 {
    assert(isFinite(prob));
    assert(0 <= prob and prob <= 1);
    assert(0 <= p and p <= 1);
    return if (p > 1 - prob) 1 else 0;
}

test "quantile.bernoulli" {
    try testing.expectApproxEqRel(bernoulli(0   , 0.2), 0, epsilon);
    try testing.expectApproxEqRel(bernoulli(0.79, 0.2), 0, epsilon);
    try testing.expectApproxEqRel(bernoulli(0.8 , 0.2), 0, epsilon);
    try testing.expectApproxEqRel(bernoulli(0.81, 0.2), 1, epsilon);
    try testing.expectApproxEqRel(bernoulli(1   , 0.2), 1, epsilon);
}

/// Quantile function of Geometric distribution
///
/// prob ∈ [0, 1]
pub fn geometric(p: f64, prob: f64) f64 {
    assert(isFinite(prob));
    assert(0 <= prob and prob <= 1);
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
    try testing.expectEqual(geometric(0  , 0), 0  );
    try testing.expectEqual(geometric(0.5, 0), inf);
    try testing.expectEqual(geometric(1  , 0), inf);

    try testing.expectApproxEqRel(geometric(0    , 0.2), 0  , epsilon);
    try testing.expectApproxEqRel(geometric(0.19 , 0.2), 0  , epsilon);
    try testing.expectApproxEqRel(geometric(0.2  , 0.2), 0  , epsilon);
    try testing.expectApproxEqRel(geometric(0.21 , 0.2), 1  , epsilon);
    try testing.expectApproxEqRel(geometric(0.35 , 0.2), 1  , epsilon);
    try testing.expectApproxEqRel(geometric(0.36 , 0.2), 1  , epsilon);
    try testing.expectApproxEqRel(geometric(0.37 , 0.2), 2  , epsilon);
    try testing.expectEqual      (geometric(1    , 0.2), inf         );
}

/// Quantile function of Poisson distribution
///
/// lambda ∈ [0, ∞]
pub fn poisson(p: f64, lambda: f64) f64 {
    assert(!isNan(lambda));
    assert(lambda >= 0);
    assert(0 <= p and p <= 1);
    if (p == 0 or lambda == 0) {
        return 0;
    }
    if (p == 1 or isInf(lambda)) {
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
    try testing.expectEqual(poisson(0  , 0  ), 0  );
    try testing.expectEqual(poisson(0.5, 0  ), 0  );
    try testing.expectEqual(poisson(1  , 0  ), 0  );
    try testing.expectEqual(poisson(0  , inf), 0  );
    try testing.expectEqual(poisson(0.5, inf), inf);
    try testing.expectEqual(poisson(1  , inf), inf);

    try testing.expectApproxEqRel(poisson(0                 , 3), 0  , epsilon);
    try testing.expectApproxEqRel(poisson(0.0497870683678638, 3), 0  , epsilon);
    try testing.expectApproxEqRel(poisson(0.0497870683678639, 3), 0  , epsilon);
    try testing.expectApproxEqRel(poisson(0.0497870683678640, 3), 1  , epsilon);
    try testing.expectApproxEqRel(poisson(0.1991482734714556, 3), 1  , epsilon);
    try testing.expectApproxEqRel(poisson(0.1991482734714557, 3), 1  , epsilon);
    try testing.expectApproxEqRel(poisson(0.1991482734714558, 3), 2  , epsilon);
    try testing.expectEqual      (poisson(1                 , 3), inf         );
}

/// Quantile function of Binomial distribution
///
/// size ∈ {0, 1, 2, ⋯}
///
/// prob ∈ [0, 1]
pub fn binomial(p: f64, size: u64, prob: f64) f64 {
    assert(isFinite(prob));
    assert(0 <= prob and prob <= 1);
    assert(0 <= p and p <= 1);
    const fsize = @as(f64, @floatFromInt(size));
    if (p == 0) {
        return 0;
    }
    if (p == 1) {
        return fsize;
    }
    if (prob == 0) {
        return 0;
    }
    if (prob == 1) {
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
    try testing.expectEqual(binomial(0  , 0 , 0.2), 0 );
    try testing.expectEqual(binomial(0.5, 0 , 0.2), 0 );
    try testing.expectEqual(binomial(1  , 0 , 0.2), 0 );
    try testing.expectEqual(binomial(0  , 10, 0  ), 0 );
    try testing.expectEqual(binomial(0.5, 10, 0  ), 0 );
    try testing.expectEqual(binomial(1  , 10, 0  ), 10);
    try testing.expectEqual(binomial(0  , 10, 1  ), 0 );
    try testing.expectEqual(binomial(0.5, 10, 1  ), 10);
    try testing.expectEqual(binomial(1  , 10, 1  ), 10);

    try testing.expectApproxEqRel(binomial(0           , 10, 0.2), 0 , epsilon);
    try testing.expectApproxEqRel(binomial(0.1073741823, 10, 0.2), 0 , epsilon);
    try testing.expectApproxEqRel(binomial(0.1073741824, 10, 0.2), 0 , epsilon);
    try testing.expectApproxEqRel(binomial(0.1073741825, 10, 0.2), 1 , epsilon);
    try testing.expectApproxEqRel(binomial(0.3758096383, 10, 0.2), 1 , epsilon);
    try testing.expectApproxEqRel(binomial(0.3758096384, 10, 0.2), 1 , epsilon);
    try testing.expectApproxEqRel(binomial(0.3758096385, 10, 0.2), 2 , epsilon);
    try testing.expectApproxEqRel(binomial(1           , 10, 0.2), 10, epsilon);
}

/// Quantile function of Negative Binomial distribution
///
/// size ∈ {0, 1, 2, ⋯}
///
/// prob ∈ [0, 1]
pub fn negativeBinomial(p: f64, size: u64, prob: f64) f64 {
    assert(isFinite(prob));
    assert(0 <= prob and prob <= 1);
    assert(0 <= p and p <= 1);
    // const fsize = @as(f64, @floatFromInt(size));
    if (p == 0 or prob == 1 or size == 0) {
        return 0;
    }
    if (p == 1 or prob == 0) {
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
    try testing.expectEqual(negativeBinomial(0  , 0 , 0.2), 0  );
    try testing.expectEqual(negativeBinomial(0.5, 0 , 0.2), 0  );
    try testing.expectEqual(negativeBinomial(1  , 0 , 0.2), 0  );
    try testing.expectEqual(negativeBinomial(0  , 10, 0  ), 0  );
    try testing.expectEqual(negativeBinomial(0.5, 10, 0  ), inf);
    try testing.expectEqual(negativeBinomial(1  , 10, 0  ), inf);
    try testing.expectEqual(negativeBinomial(0  , 10, 1  ), 0  );
    try testing.expectEqual(negativeBinomial(0.5, 10, 1  ), 0  );
    try testing.expectEqual(negativeBinomial(1  , 10, 1  ), 0  );

    try testing.expectApproxEqRel(negativeBinomial(0           , 10, 0.2), 0  , epsilon);
    try testing.expectApproxEqRel(negativeBinomial(0.0000001023, 10, 0.2), 0  , epsilon);
    try testing.expectApproxEqRel(negativeBinomial(0.0000001024, 10, 0.2), 0  , epsilon);
    try testing.expectApproxEqRel(negativeBinomial(0.0000001025, 10, 0.2), 1  , epsilon);
    try testing.expectApproxEqRel(negativeBinomial(0.0000009215, 10, 0.2), 1  , epsilon);
    try testing.expectApproxEqRel(negativeBinomial(0.0000009216, 10, 0.2), 1  , epsilon);
    try testing.expectApproxEqRel(negativeBinomial(0.0000009217, 10, 0.2), 2  , epsilon);
    try testing.expectEqual      (negativeBinomial(1           , 10, 0.2), inf         );
}

/// Quantile function of Exponential distribution
///
/// rate ∈ [0, ∞]
pub fn exponential(p: f64, rate: f64) f64 {
    assert(!isNan(rate));
    assert(rate >= 0);
    assert(0 <= p and p <= 1);
    if (rate == 0) {
        return if (p == 0) 0 else inf;
    }
    if (isInf(rate)) {
        return if (p == 1) inf else 0;
    }
    return -@log(1 - p) / rate;
}

test "quantile.exponential" {
    try testing.expectEqual(exponential(0  , 0  ), 0  );
    try testing.expectEqual(exponential(0.5, 0  ), inf);
    try testing.expectEqual(exponential(1  , 0  ), inf);
    try testing.expectEqual(exponential(0  , inf), 0  );
    try testing.expectEqual(exponential(0.5, inf), 0  );
    try testing.expectEqual(exponential(1  , inf), inf);

    try testing.expectApproxEqRel(exponential(0                 , 3), 0, epsilon);
    try testing.expectApproxEqRel(exponential(0.9502129316321360, 3), 1, epsilon);
    try testing.expectApproxEqRel(exponential(0.9975212478233336, 3), 2, epsilon);
    try testing.expectEqual      (exponential(1,                  3), inf       );
}

// pub fn weibull(p: f64, shape: f64, rate: f64) !f64 {
//     if (!isFinite(shape) or !isFinite(rate) or !isFinite(p)) {
//         return error.NonFiniteParam;
//     }
//     if (shape < 0 or rate < 0) {
//         return error.NegativeParam;
//     }
//     if (shape == 0 or shape == 0) {
//         return error.ZeroParam;
//     }
//     if (p < 0 or p > 1) {
//         return error.ProbOutside01;
//     }
//     if (p == 1) {
//         return error.OneParam;
//     }
//     return std.math.pow(f64, -@log(1 - p), 1 / shape) / rate;
// }

// test "quantile.exponential" {
//     try testing.expectEqual(weibull(0  , 0  ), 0  );
//     try testing.expectEqual(weibull(0.5, 0  ), inf);
//     try testing.expectEqual(weibull(1  , 0  ), inf);
//     try testing.expectEqual(weibull(0  , inf), 0  );
//     try testing.expectEqual(weibull(0.5, inf), 0  );
//     try testing.expectEqual(weibull(1  , inf), inf);

//     try testing.expectApproxEqRel(weibull(0                 , 3), 0, epsilon);
//     try testing.expectApproxEqRel(weibull(0.9502129316321360, 3), 1, epsilon);
//     try testing.expectApproxEqRel(weibull(0.9975212478233336, 3), 2, epsilon);
//     try testing.expectEqual      (weibull(1,                  3), inf       );
// }

// pub fn cauchy(p: f64, location: f64, scale: f64) !f64 {
//     if (!isFinite(location) or !isFinite(scale) or !isFinite(p)) {
//         return error.NonFiniteParam;
//     }
//     if (scale < 0) {
//         return error.NegativeParam;
//     }
//     if (scale == 0) {
//         return error.ZeroParam;
//     }
//     if (p < 0 or p > 1) {
//         return error.ProbOutside01;
//     }
//     if (p == 0) {
//         return error.ZeroParam;
//     }
//     if (p == 1) {
//         return error.OneParam;
//     }
//     return location + scale * @tan(std.math.pi * (p - 0.5));
// }

// pub fn logistic(p: f64, location: f64, scale: f64) !f64 {
//     if (!isFinite(location) or !isFinite(scale) or !isFinite(p)) {
//         return error.NonFiniteParam;
//     }
//     if (scale < 0) {
//         return error.NegativeParam;
//     }
//     if (scale == 0) {
//         return error.ZeroParam;
//     }
//     if (p < 0 or p > 1) {
//         return error.ProbOutside01;
//     }
//     if (p == 0) {
//         return error.ZeroParam;
//     }
//     if (p == 1) {
//         return error.OneParam;
//     }
//     return location + scale * @log(p / (1 - p));
// }

// pub fn gamma(p: f64, shape: f64, rate: f64) !f64 {
//     if (!isFinite(shape) or !isFinite(rate) or !isFinite(p)) {
//         return error.NonFiniteParam;
//     }
//     if (shape < 0 or rate < 0) {
//         return error.NegativeParam;
//     }
//     if (shape == 0 or rate == 0) {
//         return error.ZeroParam;
//     }
//     if (p < 0 or p > 1) {
//         return error.ProbOutside01;
//     }
//     if (p == 1) {
//         return error.OneParam;
//     }
//     if (p == 0) {
//         return 0;
//     }
//     return stdprob.inverseComplementedIncompleteGamma(shape, 1 - p) / rate;
// }

// pub fn chiSquared(p: f64, df: f64) !f64 {
//     return gamma(p, 0.5 * df, 0.5);
// }

// pub fn F(p: f64, df1: f64, df2: f64) !f64 {
//     if (!isFinite(df1) or !isFinite(df2) or !isFinite(p)) {
//         return error.NonFiniteParam;
//     }
//     if (df1 < 0 or df2 < 0) {
//         return error.NegativeParam;
//     }
//     if (df1 == 0 or df2 == 0) {
//         return error.ZeroParam;
//     }
//     if (p < 0 or p > 1) {
//         return error.ProbOutside01;
//     }
//     if (p == 1) {
//         return error.OneParam;
//     }
//     if (p == 0) {
//         return 0;
//     }
//     const q = try beta(1 - p, 0.5 * df1, 0.5 * df2);
//     return df2 / df1 * (1 / q - 1);
// }

// pub fn beta(p: f64, shape1: f64, shape2: f64) !f64 {
//     if (!isFinite(shape1) or !isFinite(shape2) or !isFinite(p)) {
//         return error.NonFiniteParam;
//     }
//     if (shape1 < 0 or shape2 < 0) {
//         return error.NegativeParam;
//     }
//     if (shape1 == 0 or shape2 == 0) {
//         return error.ZeroParam;
//     }
//     if (p < 0 or p > 1) {
//         return error.ProbOutside01;
//     }
//     if (p == 0 or p == 1) {
//         return p;
//     }
//     return stdprob.inverseIncompleteBeta(shape1, shape2, p);
// }

// pub fn normal(p: f64, mean: f64, sd: f64) !f64 {
//     if (!isFinite(mean) or !isFinite(sd) or !isFinite(p)) {
//         return error.NonFiniteParam;
//     }
//     if (sd < 0) {
//         return error.NegativeParam;
//     }
//     if (sd == 0) {
//         return error.ZeroParam;
//     }
//     if (p < 0 or p > 1) {
//         return error.ProbOutside01;
//     }
//     if (p == 0) {
//         return error.ZeroParam;
//     }
//     if (p == 1) {
//         return error.OneParam;
//     }
//     return mean + sd * stdprob.inverseNormalDist(p);
// }

// pub fn logNormal(p: f64, meanlog: f64, sdlog: f64) !f64 {
//     const q = try normal(p, meanlog, sdlog);
//     return @exp(q);
// }

// pub fn t(p: f64, df: f64) !f64 {
//     if (p < 0 or p > 1) {
//         return error.ProbOutside01;
//     }
//     if (p == 0) {
//         return error.ZeroParam;
//     }
//     if (p == 1) {
//         return error.OneParam;
//     }
//     if (p < 0.5) {
//         const q = try beta(2 * p, 0.5 * df, 0.5);
//         return -@sqrt(df / q - df);
//     } else {
//         const q = try beta(2 - 2 * p, 0.5 * df, 0.5);
//         return @sqrt(df / q - df);
//     }
// }
