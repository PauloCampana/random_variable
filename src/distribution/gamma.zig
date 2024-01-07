//! Gamma distribution.
//!
//! Parameters:
//!     α: `shape` ∈ (0,∞)
//!     λ: `rate`  ∈ (0,∞)

const std = @import("std");
const lgamma = @import("../thirdyparty/prob.zig").lnGamma;
const incompleteGamma = @import("../thirdyparty/prob.zig").incompleteGamma;
const inverseComplementedIncompleteGamma = @import("../thirdyparty/prob.zig").inverseComplementedIncompleteGamma;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

pub const parameters = 2;
pub const support = [2]f64 {0, inf};

/// f(x) = λ / gamma(α) (λx)^(α - 1) exp(-λx).
pub fn density(x: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x)) {
        return 0;
    }
    if (x == 0) {
        if (shape == 1) {
            return rate;
        }
        return if (shape < 1) inf else 0;
    }
    const z = rate * x;
    const num = @log(rate) + (shape - 1) * @log(z) - z;
    const den = lgamma(shape);
    return @exp(num - den);
}

/// No closed form.
pub fn probability(q: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    const z = rate * q;
    return incompleteGamma(shape, z);
}

/// No closed form.
pub fn quantile(p: f64, shape: f64, rate: f64) f64 {
    assert(isFinite(shape) and isFinite(rate));
    assert(shape > 0 and rate > 0);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return 0;
    }
    if (p == 1) {
        return inf;
    }
    const q = inverseComplementedIncompleteGamma(shape, 1 - p);
    return q / rate;
}

/// Uses George Marsaglia's rejection sampling.
/// https://dl.acm.org/doi/pdf/10.1145/358407.358414
pub const random = struct {
    pub fn implementation(generator: std.rand.Random, shape: f64, rate: f64) f64 {
        const correct = shape >= 1;
        const increment: f64 = if (correct) 0 else 1;
        const d = shape - 1.0 / 3.0 + increment;
        const c = 1 / (3 * @sqrt(d));
        const gam = while (true) {
            const z2, const v3 = while (true) {
                const z = generator.floatNorm(f64);
                const v = 1 + c * z;
                if (v > 0) {
                    break .{z * z, v * v * v};
                }
            };
            const uni = generator.float(f64);
            const cond0 = uni < 1 - 0.0331 * z2 * z2;
            if (cond0 or @log(uni) < 0.5 * z2 + d * (1 - v3 + @log(v3))) {
                break d * v3;
            }
        };
        if (correct) {
            return gam / rate;
        } else {
            const uni = generator.float(f64);
            const correction = std.math.pow(f64, uni, 1 / shape);
            return gam / rate * correction;
        }
    }

    /// shape and rate ∈ (0,∞)
    pub fn single(generator: std.rand.Random, shape: f64, rate: f64) f64 {
        assert(isFinite(shape) and isFinite(rate));
        assert(shape > 0 and rate > 0);
        return implementation(generator, shape, rate);
    }

    /// shape and rate ∈ (0,∞)
    pub fn buffer(buf: []f64, generator: std.rand.Random, shape: f64, rate: f64) []f64 {
        assert(isFinite(shape) and isFinite(rate));
        assert(shape > 0 and rate > 0);
        for (buf) |*x| {
            x.* = implementation(generator, shape, rate);
        }
        return buf;
    }

    /// shape and rate ∈ (0,∞)
    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, shape: f64, rate: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, shape, rate);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "gamma.density" {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectEqual(inf, density(0, 0.9, 5));
    try expectEqual(5  , density(0, 1  , 5));
    try expectEqual(0  , density(0, 1.1, 5));

    try expectApproxEqRel(0                 , density(0, 3, 5), eps);
    try expectApproxEqRel(0.4211216874428417, density(1, 3, 5), eps);
    try expectApproxEqRel(0.0113499824406212, density(2, 3, 5), eps);
}

test "gamma.probability" {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                 , probability(0, 3, 5), eps);
    try expectApproxEqRel(0.8753479805169189, probability(1, 3, 5), eps);
    try expectApproxEqRel(0.9972306042844884, probability(2, 3, 5), eps);
}

test "gamma.quantile" {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.3070088405289287, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.4570153808006763, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(0.6210757194526701, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(0.8558059720250668, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf               , quantile(1  , 3, 5)     );
}

test "gamma.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.c5f1fac1e8796p-2, random.implementation(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.ffa96ffd15766p-2, random.implementation(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.0ff0a4d0472aap-1, random.implementation(gen, 3, 5), eps);
}
