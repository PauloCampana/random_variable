//! Support: [0,∞)
//!
//! Parameters:
//! - α: `shape` ∈ (0,∞)
//! - σ: `scale` ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = α / σ exp(α(1 - exp(x / σ)) + x / σ)
pub fn density(x: f64, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    assert(!isNan(x));
    if (x < 0 or x == inf) {
        return 0;
    }
    const z = x / scale;
    const inner = 1 - @exp(z);
    const outer = @exp(shape * inner + z);
    return shape / scale * outer;
}

/// F(q) = 1 - exp(α(1 - exp(q / σ)))
pub fn probability(q: f64, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = q / scale;
    const inner = 1 - @exp(z);
    const outer = @exp(shape * inner);
    return 1 - outer;
}

/// Q(p) = σ ln(1 - ln(1 - p) / α)
pub fn quantile(p: f64, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    assert(0 <= p and p <= 1);
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
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    const exp = generator.floatExp(f64);
    return scale * @log(1 + exp / shape);
}

pub fn fill(buffer: []f64, generator: std.Random, shape: f64, scale: f64) void {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    for (buffer) |*x| {
        const exp = generator.floatExp(f64);
        x.* = scale * @log(1 + exp / shape);
    }
}

export fn rv_gompertz_density(x: f64, shape: f64, scale: f64) f64 {
    return density(x, shape, scale);
}
export fn rv_gompertz_probability(q: f64, shape: f64, scale: f64) f64 {
    return probability(q, shape, scale);
}
export fn rv_gompertz_quantile(p: f64, shape: f64, scale: f64) f64 {
    return quantile(p, shape, scale);
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

test quantile {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.3587242641511828, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.7861947079791208, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(1.3326633764798619, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(2.1474681650907835, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf               , quantile(1  , 3, 5)     );
}
