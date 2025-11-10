//! Support: (-∞,∞)
//!
//! Parameters:
//! - ν: `df` ∈ (0,∞)

const std = @import("std");
const assert = @import("../assert.zig");
const special = @import("../special.zig");
const gamma = @import("gamma.zig");
const inf = std.math.inf(f64);

/// f(x) (ν / (ν + x^2))^((ν + 1) / 2) / (sqrt(ν) beta(ν / 2, 1 / 2))
pub fn density(x: f64, df: f64) callconv(.c) f64 {
    assert.t(df);
    assert.real(x);

    const num = (0.5 * df + 0.5) * @log(df / (df + x * x));
    const den = 0.5 * @log(df) + special.lbeta(0.5 * df, 0.5);
    return @exp(num - den);
}

/// No closed form
pub fn probability(q: f64, df: f64) callconv(.c) f64 {
    assert.t(df);
    assert.real(q);

    if (q == inf) {
        return 1;
    }
    const z = q * q;
    if (q < 0) {
        const p = df / (df + z);
        return 0.5 * special.beta.probability(p, 0.5 * df, 0.5);
    } else {
        const p = z / (df + z);
        return 0.5 * special.beta.probability(p, 0.5, 0.5 * df) + 0.5;
    }
}

/// No closed form
pub fn survival(t: f64, df: f64) callconv(.c) f64 {
    assert.t(df);
    assert.real(t);

    if (t == -inf) {
        return 1;
    }
    const z = t * t;
    if (t > 0) {
        const p = df / (df + z);
        return 0.5 * special.beta.probability(p, 0.5 * df, 0.5);
    } else {
        const p = z / (df + z);
        return 0.5 * special.beta.probability(p, 0.5, 0.5 * df) + 0.5;
    }
}

/// No closed form
pub fn quantile(p: f64, df: f64) callconv(.c) f64 {
    assert.t(df);
    assert.probability(p);

    if (p < 0.5) {
        const q = special.beta.quantile(2 * p, 0.5 * df, 0.5);
        return -@sqrt(df / q - df);
    } else {
        const q = special.beta.quantile(2 - 2 * p, 0.5 * df, 0.5);
        return @sqrt(df / q - df);
    }
}

pub fn random(generator: std.Random, df: f64) f64 {
    assert.t(df);

    if (df == 1) {
        const uni = generator.float(f64);
        return @tan(std.math.pi * uni);
    }
    const nor = generator.floatNorm(f64);
    const chi = gamma.random(generator, 0.5 * df, 2);
    return nor * @sqrt(df / chi);
}

pub fn fill(buffer: []f64, generator: std.Random, df: f64) void {
    assert.t(df);

    if (df == 1) {
        for (buffer) |*x| {
            const uni = generator.float(f64);
            x.* = @tan(std.math.pi * uni);
        }
        return;
    }
    const hdf = 0.5 * df;
    for (buffer) |*x| {
        const nor = generator.floatNorm(f64);
        const chi = gamma.random(generator, hdf, 2);
        x.* = nor * @sqrt(df / chi);
    }
}

comptime {
    @export(&density, .{ .name = "rv_t_density" });
    @export(&probability, .{ .name = "rv_t_probability" });
    @export(&survival, .{ .name = "rv_t_survival" });
    @export(&quantile, .{ .name = "rv_t_quantile" });
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3));
    try expectEqual(0, density( inf, 3));

    try expectApproxEqRel(0.3675525969478613, density(0, 3), eps);
    try expectApproxEqRel(0.2067483357831720, density(1, 3), eps);
    try expectApproxEqRel(0.0675096606638929, density(2, 3), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3));
    try expectEqual(1, probability( inf, 3));

    try expectApproxEqRel(0.5               , probability(0, 3), eps);
    try expectApproxEqRel(0.8044988905221148, probability(1, 3), eps);
    try expectApproxEqRel(0.9303370157205784, probability(2, 3), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 3));
    try expectEqual(0, survival( inf, 3));

    try expectApproxEqRel(0.5               , survival(0, 3), eps);
    try expectApproxEqRel(0.1955011094778853, survival(1, 3), eps);
    try expectApproxEqRel(0.0696629842794215, survival(2, 3), eps);
}

test quantile {
    try expectEqual      (-inf               , quantile(0  , 3)     );
    try expectApproxEqRel(-0.9784723123633045, quantile(0.2, 3), eps);
    try expectApproxEqRel(-0.2766706623326898, quantile(0.4, 3), eps);
    try expectApproxEqRel( 0.2766706623326902, quantile(0.6, 3), eps);
    try expectApproxEqRel( 0.9784723123633039, quantile(0.8, 3), eps);
    try expectEqual      ( inf               , quantile(1  , 3)     );
}
