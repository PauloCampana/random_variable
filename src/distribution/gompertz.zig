//! Support: [0,∞)
//!
//! Parameters:
//! - α: `shape` ∈ (0,∞)
//! - σ: `scale` ∈ (0,∞)

const std = @import("std");
const assert = @import("../assert.zig");
const inf = std.math.inf(f64);

/// f(x) = α / σ exp(α(1 - exp(x / σ)) + x / σ)
pub fn density(x: f64, shape: f64, scale: f64) callconv(.c) f64 {
    assert.gompertz(shape, scale);
    assert.real(x);

    if (x < 0 or x == inf) {
        return 0;
    }
    const z = x / scale;
    const inner = -std.math.expm1(z);
    const outer = @exp(shape * inner + z);
    return shape / scale * outer;
}

/// F(q) = 1 - exp(α(1 - exp(q / σ)))
pub fn probability(q: f64, shape: f64, scale: f64) callconv(.c) f64 {
    assert.gompertz(shape, scale);
    assert.real(q);

    if (q <= 0) {
        return 0;
    }
    const z = q / scale;
    const inner = -std.math.expm1(z);
    return -std.math.expm1(shape * inner);
}

/// S(t) = exp(α(1 - exp(t / σ)))
pub fn survival(t: f64, shape: f64, scale: f64) callconv(.c) f64 {
    assert.gompertz(shape, scale);
    assert.real(t);

    if (t <= 0) {
        return 1;
    }
    const z = t / scale;
    const inner = -std.math.expm1(z);
    return @exp(shape * inner);
}

/// Q(p) = σ ln(1 - ln(1 - p) / α)
pub fn quantile(p: f64, shape: f64, scale: f64) callconv(.c) f64 {
    assert.gompertz(shape, scale);
    assert.probability(p);

    if (p == 0) {
        return 0;
    }
    if (p == 1) {
        return inf;
    }
    const inner = std.math.log1p(-p);
    const outer = std.math.log1p(-inner / shape);
    return scale * outer;
}

pub fn random(generator: std.Random, shape: f64, scale: f64) f64 {
    assert.gompertz(shape, scale);

    const exp = generator.floatExp(f64);
    return scale * @log(1 + exp / shape);
}

pub fn fill(buffer: []f64, generator: std.Random, shape: f64, scale: f64) void {
    assert.gompertz(shape, scale);

    for (buffer) |*x| {
        const exp = generator.floatExp(f64);
        x.* = scale * @log(1 + exp / shape);
    }
}

comptime {
    @export(&density, .{ .name = "rv_gompertz_density" });
    @export(&probability, .{ .name = "rv_gompertz_probability" });
    @export(&survival, .{ .name = "rv_gompertz_survival" });
    @export(&quantile, .{ .name = "rv_gompertz_quantile" });
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectApproxEqRel(0.6               , density(0, 3, 5), eps);
    try expectApproxEqRel(0.3771795676204895, density(1, 3, 5), eps);
    try expectApproxEqRel(0.2046815920799842, density(2, 3, 5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                 , probability(0, 3, 5), eps);
    try expectApproxEqRel(0.4853191475940816, probability(1, 3, 5), eps);
    try expectApproxEqRel(0.7713297096238283, probability(2, 3, 5), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 3, 5));
    try expectEqual(0, survival( inf, 3, 5));

    try expectApproxEqRel(1                 , survival(0, 3, 5), eps);
    try expectApproxEqRel(0.5146808524059183, survival(1, 3, 5), eps);
    try expectApproxEqRel(0.2286702903761716, survival(2, 3, 5), eps);
}

test quantile {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.3587242641511828, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.7861947079791208, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(1.3326633764798619, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(2.1474681650907835, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf               , quantile(1  , 3, 5)     );
}
