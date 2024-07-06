//! Support: [0,∞)
//!
//! Parameters:
//! - n: `df1` ∈ (0,∞)
//! - m: `df2` ∈ (0,∞)

const std = @import("std");
const betaPrime = @import("betaPrime.zig");
const special = @import("../special.zig");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = n^(n / 2) m^(m / 2) x^(n / 2 - 1) (m + nx)^(-(n + m) / 2) / beta(n / 2, m / 2)
pub fn density(x: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(!isNan(x));
    if (x < 0 or x == inf) {
        return 0;
    }
    if (x == 0) {
        if (df1 == 2) {
            return 1;
        }
        return if (df1 < 2) inf else 0;
    }
    const hdf1 = 0.5 * df1;
    const hdf2 = 0.5 * df2;
    const num1 = hdf1 * @log(df1) + hdf2 * @log(df2) + (hdf1 - 1) * @log(x);
    const num2 = -(hdf1 + hdf2) * @log(df2 + df1 * x);
    const den = special.lbeta(hdf1, hdf2);
    return @exp(num1 + num2 - den);
}

/// No closed form
pub fn probability(q: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    if (q == inf) {
        return 1;
    }
    const z = df1 * q;
    const p = z / (df2 + z);
    return special.beta_probability(0.5 * df1, 0.5 * df2, p);
}

/// No closed form
pub fn survival(t: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(!isNan(t));
    if (t <= 0) {
        return 1;
    }
    const z = df1 * t;
    const p = df2 / (df2 + z);
    return special.beta_probability(0.5 * df2, 0.5 * df1, p);
}

/// No closed form
pub fn quantile(p: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(0 <= p and p <= 1);
    const q = special.beta_quantile(0.5 * df2, 0.5 * df1, 1 - p);
    return (df2 / q - df2) / df1;
}

pub fn random(generator: std.Random, df1: f64, df2: f64) f64 {
    const bp = betaPrime.random(generator, 0.5 * df1, 0.5 * df2);
    return df2 / df1 * bp;
}

pub fn fill(buffer: []f64, generator: std.Random, df1: f64, df2: f64) void {
    const hdf1 = 0.5 * df1;
    const hdf2 = 0.5 * df2;
    const ratio = df2 / df1;
    for (buffer) |*x| {
        const bp = betaPrime.random(generator, hdf1, hdf2);
        x.* = ratio * bp;
    }
}

export fn rv_f_density(x: f64, df1: f64, df2: f64) f64 {
    return density(x, df1, df2);
}
export fn rv_f_probability(q: f64, df1: f64, df2: f64) f64 {
    return probability(q, df1, df2);
}
export fn rv_f_survival(t: f64, df1: f64, df2: f64) f64 {
    return survival(t, df1, df2);
}
export fn rv_f_quantile(p: f64, df1: f64, df2: f64) f64 {
    return quantile(p, df1, df2);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectEqual(inf, density(0, 1.8, 5));
    try expectEqual(  1, density(0, 2  , 5));
    try expectEqual(  0, density(0, 2.2, 5));

    try expectApproxEqRel(0                 , density(0, 3, 5), eps);
    try expectApproxEqRel(0.3611744789422851, density(1, 3, 5), eps);
    try expectApproxEqRel(0.1428963909075316, density(2, 3, 5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                 , probability(0, 3, 5), eps);
    try expectApproxEqRel(0.5351452100063649, probability(1, 3, 5), eps);
    try expectApproxEqRel(0.7673760819999214, probability(2, 3, 5), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 3, 5));
    try expectEqual(0, survival( inf, 3, 5));

    try expectApproxEqRel(1                 , survival(0, 3, 5), eps);
    try expectApproxEqRel(0.4648547899936350, survival(1, 3, 5), eps);
    try expectApproxEqRel(0.2326239180000785, survival(2, 3, 5), eps);
}

test quantile {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.3372475270245997, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.6821342707772098, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(1.1978047828924259, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(2.2530173716474851, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf               , quantile(1  , 3, 5)     );
}
