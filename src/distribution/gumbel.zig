//! Support: (-∞,∞)
//!
//! Parameters:
//! - μ: `location` ∈ (-∞,∞)
//! - σ: `scale`    ∈ ( 0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = exp(-(x - μ) / σ - exp(-(x - μ) / σ)) / σ
pub fn density(x: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    if (x == -inf) {
        return 0;
    }
    const z = (x - location) / scale;
    const inner = @exp(-z);
    const outer = @exp(-z - inner);
    return outer / scale;
}

/// F(q) = exp(-exp(-(q - μ) / σ))
pub fn probability(q: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(q));
    const z = (q - location) / scale;
    const inner = @exp(-z);
    const outer = @exp(-inner);
    return outer;
}

/// Q(p) = μ - σ ln(-ln(p))
pub fn quantile(p: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(0 <= p and p <= 1);
    const inner = -@log(p);
    const outer = -@log(inner);
    return location + scale * outer;
}

pub fn random(generator: std.Random, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    const exp = generator.floatExp(f64);
    return location - scale * @log(exp);
}

pub fn fill(buffer: []f64, generator: std.Random, location: f64, scale: f64) void {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    for (buffer) |*x| {
        const exp = generator.floatExp(f64);
        x.* = location - scale * @log(exp);
    }
}

export fn rv_gumbel_density(x: f64, location: f64, scale: f64) f64 {
    return density(x, location, scale);
}
export fn rv_gumbel_probability(q: f64, location: f64, scale: f64) f64 {
    return probability(q, location, scale);
}
export fn rv_gumbel_quantile(p: f64, location: f64, scale: f64) f64 {
    return quantile(p, location, scale);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 0,  1));
    try expectEqual(0, density( inf, 0,  1));

    try expectApproxEqRel(0.3678794411714423, density( 0, 0,  1), eps);
    try expectApproxEqRel(0.2546463800435824, density( 1, 0,  1), eps);
    try expectApproxEqRel(0.1182049515931431, density( 2, 0,  1), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0.3678794411714423, probability( 0, 0,  1), eps);
    try expectApproxEqRel(0.6922006275553463, probability( 1, 0,  1), eps);
    try expectApproxEqRel(0.8734230184931166, probability( 2, 0,  1), eps);
}

test quantile {
    try expectEqual      (-inf               , quantile(0  , 0, 1)     );
    try expectApproxEqRel(-0.4758849953271106, quantile(0.2, 0, 1), eps);
    try expectApproxEqRel( 0.0874215717907550, quantile(0.4, 0, 1), eps);
    try expectApproxEqRel( 0.6717269920921220, quantile(0.6, 0, 1), eps);
    try expectApproxEqRel( 1.4999399867595155, quantile(0.8, 0, 1), eps);
    try expectEqual      ( inf               , quantile(1  , 0, 1)     );
}
