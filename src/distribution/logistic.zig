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

/// f(x) = exp(-(x - μ) / σ) / (σ (1 + exp(-(x - μ) / σ))^2)
pub fn density(x: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(x));
    const z = @abs(x - location) / scale;
    const expz = @exp(-z);
    const expzp1 = expz + 1;
    return expz / (scale * expzp1 * expzp1);
}

/// F(q) = 1 / (1 + exp(-(q - μ) / σ))
pub fn probability(q: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(q));
    const z = (q - location) / scale;
    return 1 / (1 + @exp(-z));
}

/// S(t) = 1 / (1 + exp((t - μ) / σ))
pub fn survival(t: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(!isNan(t));
    if (t == -inf) {
        return 1;
    }
    const z = (t - location) / scale;
    return 1 / (1 + @exp(z));
}

/// Q(p) = μ + σ ln(p / (1 - p))
pub fn quantile(p: f64, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    assert(0 <= p and p <= 1);
    const q = @log(p / (1 - p));
    return location + scale * q;
}

pub fn random(generator: std.Random, location: f64, scale: f64) f64 {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    const uni = generator.float(f64);
    return location + scale * @log(uni / (1 - uni));
}

pub fn fill(buffer: []f64, generator: std.Random, location: f64, scale: f64) void {
    assert(isFinite(location) and isFinite(scale));
    assert(scale > 0);
    for (buffer) |*x| {
        const uni = generator.float(f64);
        x.* = location + scale * @log(uni / (1 - uni));
    }
}

export fn rv_logistic_density(x: f64, location: f64, scale: f64) f64 {
    return density(x, location, scale);
}
export fn rv_logistic_probability(q: f64, location: f64, scale: f64) f64 {
    return probability(q, location, scale);
}
export fn rv_logistic_survival(t: f64, location: f64, scale: f64) f64 {
    return survival(t, location, scale);
}
export fn rv_logistic_quantile(p: f64, location: f64, scale: f64) f64 {
    return quantile(p, location, scale);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 0, 1));
    try expectEqual(0, density( inf, 0, 1));

    try expectApproxEqRel(0.25              , density(0, 0, 1), eps);
    try expectApproxEqRel(0.1966119332414819, density(1, 0, 1), eps);
    try expectApproxEqRel(0.1049935854035065, density(2, 0, 1), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 0, 1));
    try expectEqual(1, probability( inf, 0, 1));

    try expectApproxEqRel(0.5               , probability(0, 0, 1), eps);
    try expectApproxEqRel(0.7310585786300049, probability(1, 0, 1), eps);
    try expectApproxEqRel(0.8807970779778823, probability(2, 0, 1), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 0, 1));
    try expectEqual(0, survival( inf, 0, 1));

    try expectApproxEqRel(0.5               , survival(0, 0, 1), eps);
    try expectApproxEqRel(0.2689414213699951, survival(1, 0, 1), eps);
    try expectApproxEqRel(0.1192029220221175, survival(2, 0, 1), eps);
}

test quantile {
    try expectEqual      (-inf               , quantile(0  , 0, 1)     );
    try expectApproxEqRel(-1.3862943611198906, quantile(0.2, 0, 1), eps);
    try expectApproxEqRel(-0.4054651081081643, quantile(0.4, 0, 1), eps);
    try expectApproxEqRel( 0.4054651081081648, quantile(0.6, 0, 1), eps);
    try expectApproxEqRel( 1.3862943611198908, quantile(0.8, 0, 1), eps);
    try expectEqual      ( inf               , quantile(1  , 0, 1)     );
}
