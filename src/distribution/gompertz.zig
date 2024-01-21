//! Support: X ∈ [0,∞)
//!
//! Parameters:
//! - ξ: `shape` ∈ (0,∞)
//! - σ: `scale` ∈ (0,∞)

const std = @import("std");
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const discrete = false;
pub const parameters = 2;

/// f(x) = ξ / σ exp(ξ(1 - exp(x / σ)) + x / σ).
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

/// F(q) = 1 - exp(ξ(1 - exp(q / σ))).
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

/// Q(p) = σ ln(1 - ln(1 - p) / ξ).
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

/// Uses the relation to Exponential distribution.
pub const random = struct {
    pub fn single(generator: std.rand.Random, shape: f64, scale: f64) f64 {
        assert(isFinite(shape) and isFinite(scale));
        assert(shape > 0 and scale > 0);
        const exp = generator.floatExp(f64);
        return scale * @log(1 + exp / shape);
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, shape: f64, scale: f64) []f64 {
        assert(isFinite(shape) and isFinite(scale));
        assert(shape > 0 and scale > 0);
        for (buffer) |*x| {
            const exp = generator.floatExp(f64);
            x.* = scale * @log(1 + exp / shape);
        }
        return buffer;
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "gompertz.density" {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectApproxEqRel(0.6               , density(0, 3, 5), eps);
    try expectApproxEqRel(0.3771795676204895, density(1, 3, 5), eps);
    try expectApproxEqRel(0.2046815920799842, density(2, 3, 5), eps);
}

test "gompertz.probability" {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                 , probability(0, 3, 5), eps);
    try expectApproxEqRel(0.4853191475940816, probability(1, 3, 5), eps);
    try expectApproxEqRel(0.7713297096238283, probability(2, 3, 5), eps);
}

test "gompertz.quantile" {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.3587242641511828, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.7861947079791208, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(1.3326633764798619, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(2.1474681650907835, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf               , quantile(1  , 3, 5)     );
}

test "gompertz.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.45be6394cc4b3p-2, random.single(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.531262f81efeep+1, random.single(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.916d1822686ddp-4, random.single(gen, 3, 5), eps);
}
