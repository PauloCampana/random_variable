//! Support: [0,∞)
//!
//! Parameters:
//! - α: `shape` ∈ (0,∞)
//! - σ: `scale` ∈ (0,∞)

const std = @import("std");
const special = @import("../special.zig");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// f(x) = 1 / (σ gamma(α)) (x / σ)^(α - 1) exp(-x / σ)
pub fn density(x: f64, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    assert(!isNan(x));
    if (x < 0 or x == inf) {
        return 0;
    }
    if (x == 0) {
        if (shape == 1) {
            return 1 / scale;
        }
        return if (shape < 1) inf else 0;
    }
    const z = x / scale;
    const num = (shape - 1) * @log(z) - z;
    const den = std.math.lgamma(f64, shape) + @log(scale);
    return @exp(num - den);
}

/// No closed form
pub fn probability(q: f64, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    assert(!isNan(q));
    const z = q / scale;
    return special.gamma_probability(shape, z);
}

/// No closed form
pub fn survival(t: f64, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    assert(!isNan(t));
    const z = t / scale;
    return special.gamma_survival(shape, z);
}

/// No closed form
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
    const q = special.gamma_quantile_mirrored(shape, 1 - p);
    return scale * q;
}

pub fn random(generator: std.Random, shape: f64, scale: f64) f64 {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    if (shape == 1) {
        const exp = generator.floatExp(f64);
        return scale * exp;
    }
    const correct = shape >= 1;
    const increment: f64 = if (correct) 0 else 1;
    const d = shape - 1.0 / 3.0 + increment;
    const c = 1 / (3 * @sqrt(d));
    const gam = rejection(generator, d, c);
    if (correct) {
        return scale * gam;
    }
    const uni = generator.float(f64);
    const fix = std.math.pow(f64, uni, 1 / shape);
    return scale * fix * gam;
}

pub fn fill(buffer: []f64, generator: std.Random, shape: f64, scale: f64) void {
    assert(isFinite(shape) and isFinite(scale));
    assert(shape > 0 and scale > 0);
    if (shape == 1) {
        for (buffer) |*x| {
            const exp = generator.floatExp(f64);
            x.* = scale * exp;
        }
        return;
    }
    const invshape = 1 / shape;
    const correct = shape >= 1;
    const increment: f64 = if (correct) 0 else 1;
    const d = shape - 1.0 / 3.0 + increment;
    const c = 1 / (3 * @sqrt(d));
    for (buffer) |*x| {
        const gam = rejection(generator, d, c);
        if (correct) {
            x.* = scale * gam;
            continue;
        }
        const uni = generator.float(f64);
        const fix = std.math.pow(f64, uni, invshape);
        x.* = scale * fix * gam;
    }
}

/// https://dl.acm.org/doi/pdf/10.1145/358407.358414
fn rejection(generator: std.Random, d: f64, c: f64) f64 {
    return while (true) {
        const z2, const v3 = while (true) {
            const z = generator.floatNorm(f64);
            const v = 1 + c * z;
            if (v > 0) {
                break .{ z * z, v * v * v };
            }
        };
        const uni = generator.float(f64);
        if (uni < 1 - 0.0331 * z2 * z2) {
            break d * v3;
        }
        if (@log(uni) < 0.5 * z2 + d * (1 - v3 + @log(v3))) {
            break d * v3;
        }
    };
}

export fn rv_gamma_density(x: f64, shape: f64, scale: f64) f64 {
    return density(x, shape, scale);
}
export fn rv_gamma_probability(q: f64, shape: f64, scale: f64) f64 {
    return probability(q, shape, scale);
}
export fn rv_gamma_survival(t: f64, shape: f64, scale: f64) f64 {
    return survival(t, shape, scale);
}
export fn rv_gamma_quantile(p: f64, shape: f64, scale: f64) f64 {
    return quantile(p, shape, scale);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectEqual(inf, density(0, 0.9, 5));
    try expectEqual(0.2, density(0, 1  , 5));
    try expectEqual(0  , density(0, 1.1, 5));

    try expectApproxEqRel(0                   , density(0, 3, 5), eps);
    try expectApproxEqRel(0.003274923012311927, density(1, 3, 5), eps);
    try expectApproxEqRel(0.010725120736570228, density(2, 3, 5), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                   , probability(0, 3, 5), eps);
    try expectApproxEqRel(0.001148481244862132, probability(1, 3, 5), eps);
    try expectApproxEqRel(0.007926331867253834, probability(2, 3, 5), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 3, 5));
    try expectEqual(0, survival( inf, 3, 5));

    try expectApproxEqRel(1                 , survival(0, 3, 5), eps);
    try expectApproxEqRel(0.9988515187551378, survival(1, 3, 5), eps);
    try expectApproxEqRel(0.9920736681327461, survival(2, 3, 5), eps);
}

test quantile {
    try expectApproxEqRel( 0               , quantile(0  , 3, 5), eps);
    try expectApproxEqRel( 7.67522101322321, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(11.42538452001690, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(15.52689298631674, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(21.39514930062666, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf              , quantile(1  , 3, 5)     );
}
