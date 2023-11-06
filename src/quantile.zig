const std = @import("std");
const stdprob = @import("thirdyparty/prob.zig");
const density = @import("density.zig");
const isFinite = std.math.isFinite;

pub fn uniform(p: f64, min: f64, max: f64) !f64 {
    if (!isFinite(min) or !isFinite(max) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    return min + (max - min) * p;
}

pub fn bernoulli(p: f64, prob: f64) !bool {
    if (!isFinite(prob) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (prob < 0 or prob > 1) {
        return error.ProbOutside01;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    return p > 1 - prob;
}

pub fn geometric(p: f64, prob: f64) !u64 {
    if (!isFinite(prob) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (prob < 0 or prob > 1) {
        return error.ProbOutside01;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p < prob) {
        return 0;
    }
    const q = @floor(@log(1 - p) / std.math.log1p(-prob) - 1e-8);
    return @as(u64, @intFromFloat(q));
}

pub fn poisson(p: f64, lambda: f64) !u64 {
    if (!isFinite(lambda) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (lambda < 0) {
        return error.NegativeParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 1) {
        return error.OneParam;
    }
    var q: u64 = 0;
    var sum: f64 = 0;
    while (true) : (q += 1) {
        sum += try density.poisson(q, lambda);
        if (sum > p) {
            return q;
        }
    }
}

pub fn binomial(p: f64, size: u64, prob: f64) !u64 {
    if (!isFinite(prob) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (prob < 0 or prob > 1) {
        return error.ProbOutside01;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 1) {
        return size;
    }
    var q: u64 = 0;
    var sum: f64 = 0;
    while (true) : (q += 1) {
        sum += try density.binomial(q, size, prob);
        if (sum > p) {
            return q;
        }
    }
}

pub fn negativeBinomial(p: f64, size: u64, prob: f64) !u64 {
    if (!isFinite(prob) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (prob < 0 or prob > 1) {
        return error.ProbOutside01;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 1) {
        return error.OneParam;
    }
    var q: u64 = 0;
    var sum: f64 = 0;
    while (true) : (q += 1) {
        sum += try density.negativeBinomial(q, size, prob);
        if (sum > p) {
            return q;
        }
    }
}

pub fn exponential(p: f64, rate: f64) !f64 {
    if (!isFinite(rate) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (rate < 0) {
        return error.NegativeParam;
    }
    if (rate == 0) {
        return error.ZeroParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 1) {
        return error.OneParam;
    }
    return -@log(1 - p) / rate;
}

pub fn weibull(p: f64, shape: f64, rate: f64) !f64 {
    if (!isFinite(shape) or !isFinite(rate) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (shape < 0 or rate < 0) {
        return error.NegativeParam;
    }
    if (shape == 0 or shape == 0) {
        return error.ZeroParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 1) {
        return error.OneParam;
    }
    return std.math.pow(f64, -@log(1 - p), 1 / shape) / rate;
}

pub fn cauchy(p: f64, location: f64, scale: f64) !f64 {
    if (!isFinite(location) or !isFinite(scale) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (scale < 0) {
        return error.NegativeParam;
    }
    if (scale == 0) {
        return error.ZeroParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 0) {
        return error.ZeroParam;
    }
    if (p == 1) {
        return error.OneParam;
    }
    return location + scale * @tan(std.math.pi * (p - 0.5));
}

pub fn logistic(p: f64, location: f64, scale: f64) !f64 {
    if (!isFinite(location) or !isFinite(scale) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (scale < 0) {
        return error.NegativeParam;
    }
    if (scale == 0) {
        return error.ZeroParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 0) {
        return error.ZeroParam;
    }
    if (p == 1) {
        return error.OneParam;
    }
    return location + scale * @log(p / (1 - p));
}

pub fn gamma(p: f64, shape: f64, rate: f64) !f64 {
    if (!isFinite(shape) or !isFinite(rate) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (shape < 0 or rate < 0) {
        return error.NegativeParam;
    }
    if (shape == 0 or rate == 0) {
        return error.ZeroParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 1) {
        return error.OneParam;
    }
    if (p == 0) {
        return 0;
    }
    return stdprob.inverseComplementedIncompleteGamma(shape, 1 - p) / rate;
}

pub fn chiSquared(p: f64, df: f64) !f64 {
    return gamma(p, 0.5 * df, 0.5);
}

pub fn F(p: f64, df1: f64, df2: f64) !f64 {
    if (!isFinite(df1) or !isFinite(df2) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (df1 < 0 or df2 < 0) {
        return error.NegativeParam;
    }
    if (df1 == 0 or df2 == 0) {
        return error.ZeroParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 1) {
        return error.OneParam;
    }
    if (p == 0) {
        return 0;
    }
    const q = try beta(1 - p, 0.5 * df1, 0.5 * df2);
    return df2 / df1 * (1 / q - 1);
}

pub fn beta(p: f64, shape1: f64, shape2: f64) !f64 {
    if (!isFinite(shape1) or !isFinite(shape2) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (shape1 < 0 or shape2 < 0) {
        return error.NegativeParam;
    }
    if (shape1 == 0 or shape2 == 0) {
        return error.ZeroParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 0 or p == 1) {
        return p;
    }
    return stdprob.inverseIncompleteBeta(shape1, shape2, p);
}

pub fn normal(p: f64, mean: f64, sd: f64) !f64 {
    if (!isFinite(mean) or !isFinite(sd) or !isFinite(p)) {
        return error.NonFiniteParam;
    }
    if (sd < 0) {
        return error.NegativeParam;
    }
    if (sd == 0) {
        return error.ZeroParam;
    }
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 0) {
        return error.ZeroParam;
    }
    if (p == 1) {
        return error.OneParam;
    }
    return mean + sd * stdprob.inverseNormalDist(p);
}

pub fn lognormal(p: f64, meanlog: f64, sdlog: f64) !f64 {
    const q = try normal(p, meanlog, sdlog);
    return @exp(q);
}

pub fn t(p: f64, df: f64) !f64 {
    if (p < 0 or p > 1) {
        return error.ProbOutside01;
    }
    if (p == 0) {
        return error.ZeroParam;
    }
    if (p == 1) {
        return error.OneParam;
    }
    if (p < 0.5) {
        const q = try beta(2 * p, 0.5 * df, 0.5);
        return -@sqrt(df / q - 1);
    } else {
        const q = try beta(2 - 2 * p, 0.5 * df, 0.5);
        return @sqrt(df / q - 1);
    }
}
