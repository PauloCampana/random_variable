const std = @import("std");

pub fn lbinomial(n: f64, k: f64) f64 {
    const num = std.math.lgamma(f64, n + 1);
    const den1 = std.math.lgamma(f64, k + 1);
    const den2 = std.math.lgamma(f64, n - k + 1);
    return num - den1 - den2;
}

pub fn binomial(n: f64, k: f64) f64 {
    return @exp(lbinomial(n, k));
}

pub fn lbeta(a: f64, b: f64) f64 {
    const num1 = std.math.lgamma(f64, a);
    const num2 = std.math.lgamma(f64, b);
    const den = std.math.lgamma(f64, a + b);
    return num1 + num2 - den;
}

pub fn beta(a: f64, b: f64) f64 {
    return @exp(lbeta(a, b));
}
