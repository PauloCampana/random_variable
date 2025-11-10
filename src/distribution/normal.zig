//! Support: (-∞,∞)
//!
//! Parameters:
//! - μ: `location` ∈ (-∞,∞)
//! - σ: `scale`    ∈ ( 0,∞)

const std = @import("std");
const assert = @import("../assert.zig");
const special = @import("../special.zig");
const inf = std.math.inf(f64);

/// f(x) = exp(-((x - μ) / σ)^2 / 2) / (σ sqrt(2π))
pub fn density(x: f64, location: f64, scale: f64) callconv(.c) f64 {
    assert.normal(location, scale);
    assert.real(x);

    const z = (x - location) / scale;
    const sqrt2pi = comptime @sqrt(2 * std.math.pi);
    return @exp(-0.5 * z * z) / (scale * sqrt2pi);
}

/// No closed form
pub fn probability(q: f64, location: f64, scale: f64) callconv(.c) f64 {
    assert.normal(location, scale);
    assert.real(q);

    const z = (q - location) / scale;
    return special.normal.probability(z);
}

/// No closed form
pub fn survival(t: f64, location: f64, scale: f64) callconv(.c) f64 {
    assert.normal(location, scale);
    assert.real(t);

    const z = (t - location) / scale;
    return special.normal.survival(z);
}

/// No closed form
pub fn quantile(p: f64, location: f64, scale: f64) callconv(.c) f64 {
    assert.normal(location, scale);
    assert.probability(p);

    const q = special.normal.quantile(p);
    return location + scale * q;
}

pub fn random(generator: std.Random, location: f64, scale: f64) f64 {
    assert.normal(location, scale);

    const nor = generator.floatNorm(f64);
    return location + scale * nor;
}

pub fn fill(buffer: []f64, generator: std.Random, location: f64, scale: f64) void {
    assert.normal(location, scale);

    for (buffer) |*x| {
        const nor = generator.floatNorm(f64);
        x.* = location + scale * nor;
    }
}

comptime {
    @export(&density, .{ .name = "rv_normal_density" });
    @export(&probability, .{ .name = "rv_normal_probability" });
    @export(&survival, .{ .name = "rv_normal_survival" });
    @export(&quantile, .{ .name = "rv_normal_quantile" });
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 0, 1));
    try expectEqual(0, density( inf, 0, 1));

    try expectApproxEqRel(0.3989422804014327, density(0, 0, 1), eps);
    try expectApproxEqRel(0.2419707245191433, density(1, 0, 1), eps);
    try expectApproxEqRel(0.0539909665131880, density(2, 0, 1), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0.5               , probability(0, 0, 1), eps);
    try expectApproxEqRel(0.8413447460685429, probability(1, 0, 1), eps);
    try expectApproxEqRel(0.9772498680518208, probability(2, 0, 1), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 0, 1));
    try expectEqual(0, survival( inf, 0, 1));

    try expectApproxEqRel(0.5                , survival(0, 0, 1), eps);
    try expectApproxEqRel(0.15865525393145705, survival(1, 0, 1), eps);
    try expectApproxEqRel(0.02275013194817920, survival(2, 0, 1), eps);
}

test quantile {
    try expectEqual      (-inf               , quantile(0  , 0, 1)     );
    try expectApproxEqRel(-0.8416212335729142, quantile(0.2, 0, 1), eps);
    try expectApproxEqRel(-0.2533471031357998, quantile(0.4, 0, 1), eps);
    try expectApproxEqRel( 0.2533471031358001, quantile(0.6, 0, 1), eps);
    try expectApproxEqRel( 0.8416212335729144, quantile(0.8, 0, 1), eps);
    try expectEqual      ( inf               , quantile(1  , 0, 1)     );
}
