const std = @import("std");
const assert = std.debug.assert;

pub fn real(x: f64) void {
    assert(!std.math.isNan(x));
}

pub fn probability(x: f64) void {
    assert(0 <= x and x <= 1);
}

pub fn benford(base: u64) void {
    assert(base >= 2);
}

pub fn bernoulli(prob: f64) void {
    assert(0 <= prob and prob <= 1);
}

pub fn beta_binomial(size: u64, shape1: f64, shape2: f64) void {
    _ = size;
    assert(std.math.isFinite(shape1) and shape1 > 0);
    assert(std.math.isFinite(shape2) and shape2 > 0);
}

pub fn beta_prime(shape1: f64, shape2: f64) void {
    assert(std.math.isFinite(shape1) and shape1 > 0);
    assert(std.math.isFinite(shape2) and shape2 > 0);
}

pub fn beta(shape1: f64, shape2: f64) void {
    assert(std.math.isFinite(shape1) and shape1 > 0);
    assert(std.math.isFinite(shape2) and shape2 > 0);
}

pub fn binomial(size: u64, prob: f64) void {
    _ = size;
    assert(0 <= prob and prob <= 1);
}

pub fn cauchy(location: f64, scale: f64) void {
    assert(std.math.isFinite(location));
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn chi(df: f64) void {
    assert(std.math.isFinite(df) and df > 0);
}

pub fn continuous_bernoulli(shape: f64) void {
    assert(0 < shape and shape < 1);
}

pub fn dagum(shape1: f64, shape2: f64, scale: f64) void {
    assert(std.math.isFinite(shape1) and shape1 > 0);
    assert(std.math.isFinite(shape2) and shape2 > 0);
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn discrete_uniform(min: i64, max: i64) void {
    assert(min <= max);
}

pub fn exponential(scale: f64) void {
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn f(df1: f64, df2: f64) void {
    assert(std.math.isFinite(df1) and df1 > 0);
    assert(std.math.isFinite(df2) and df2 > 0);
}

pub fn gamma(shape: f64, scale: f64) void {
    assert(std.math.isFinite(shape) and shape > 0);
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn geometric(prob: f64) void {
    assert(0 < prob and prob <= 1);
}

pub fn gompertz(shape: f64, scale: f64) void {
    assert(std.math.isFinite(shape) and shape > 0);
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn gumbel(location: f64, scale: f64) void {
    assert(std.math.isFinite(location));
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn hypergeometric(total: u64, good: u64, tries: u64) void {
    assert(good <= total);
    assert(tries <= total);
}

pub fn laplace(location: f64, scale: f64) void {
    assert(std.math.isFinite(location));
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn log_normal(location: f64, scale: f64) void {
    assert(std.math.isFinite(location));
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn logarithmic(prob: f64) void {
    assert(0 < prob and prob < 1);
}

pub fn logistic(location: f64, scale: f64) void {
    assert(std.math.isFinite(location));
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn negative_binomial(size: u64, prob: f64) void {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
}

pub fn normal(location: f64, scale: f64) void {
    assert(std.math.isFinite(location));
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn pareto(shape: f64, minimum: f64) void {
    assert(std.math.isFinite(shape) and shape > 0);
    assert(std.math.isFinite(minimum) and minimum > 0);
}

pub fn poisson(lambda: f64) void {
    assert(std.math.isFinite(lambda) and lambda > 0);
}

pub fn rayleigh(scale: f64) void {
    assert(std.math.isFinite(scale) and scale > 0);
}

pub fn t(df: f64) void {
    assert(std.math.isFinite(df) and df > 0);
}

pub fn uniform(min: f64, max: f64) void {
    assert(std.math.isFinite(min));
    assert(std.math.isFinite(max));
    assert(min <= max);
}

pub fn weibull(shape: f64, scale: f64) void {
    assert(std.math.isFinite(shape) and shape > 0);
    assert(std.math.isFinite(scale) and scale > 0);
}
