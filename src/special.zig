// Ported from Cephes' cprob module, MIT licensed
// <https://netlib.org/cephes/>

const std = @import("std");
const prob = @import("thirdyparty/prob.zig");

pub const normal = @import("special/normal.zig");
pub const gamma = @import("special/gamma.zig");

pub const beta_probability = prob.incompleteBeta;
pub const beta_quantile = prob.inverseIncompleteBeta;

pub fn lbinomial(n: f64, k: f64) f64 {
    const num = std.math.lgamma(f64, n + 1);
    const den1 = std.math.lgamma(f64, k + 1);
    const den2 = std.math.lgamma(f64, n - k + 1);
    return num - den1 - den2;
}

pub fn lbeta(a: f64, b: f64) f64 {
    const num1 = std.math.lgamma(f64, a);
    const num2 = std.math.lgamma(f64, b);
    const den = std.math.lgamma(f64, a + b);
    return num1 + num2 - den;
}
