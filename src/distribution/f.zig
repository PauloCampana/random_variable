//! Support: X ∈ [0,∞)
//!
//! Parameters:
//! - n: `df1` ∈ (0,∞)
//! - m: `df2` ∈ (0,∞)

const std = @import("std");
const gamma = @import("gamma.zig");
const math = @import("../math.zig");
const incompleteBeta = @import("../thirdyparty/prob.zig").incompleteBeta;
const inverseIncompleteBeta = @import("../thirdyparty/prob.zig").inverseIncompleteBeta;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

pub const discrete = false;
pub const parameters = 2;

/// f(x) = n^(n / 2) m^(m / 2) x^(n / 2 - 1) (m + nx)^(-(n + m) / 2) / beta(n / 2, m / 2).
pub fn density(x: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x)) {
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
    const den = math.lbeta(hdf1, hdf2);
    return @exp(num1 + num2 - den);
}

/// No closed form.
pub fn probability(q: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(!isNan(q));
    if (q <= 0) {
        return 0;
    }
    if (isInf(q)) {
        return 1;
    }
    const z = df1 * q;
    const p = z / (df2 + z);
    return incompleteBeta(0.5 * df1, 0.5 * df2, p);
}

/// No closed form.
pub fn quantile(p: f64, df1: f64, df2: f64) f64 {
    assert(isFinite(df1) and isFinite(df2));
    assert(df1 > 0 and df2 > 0);
    assert(0 <= p and p <= 1);
    const q = inverseIncompleteBeta(0.5 * df2, 0.5 * df1, 1 - p);
    return (df2 / q - df2) / df1;
}

/// Uses the relation to Gamma distribution.
pub const random = struct {
    pub fn single(generator: std.rand.Random, df1: f64, df2: f64) f64 {
        const num = gamma.random.single(generator, 0.5 * df1, 1);
        const den = gamma.random.single(generator, 0.5 * df2, 1);
        return num / den * df2 / df1;
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, df1: f64, df2: f64) []f64 {
        const hdf1 = 0.5 * df1;
        const hdf2 = 0.5 * df2;
        const ratio = df2 / df1;
        for (buffer) |*x| {
            const num = gamma.random.single(generator, hdf1, 1);
            const den = gamma.random.single(generator, hdf2, 1);
            x.* = num / den * ratio;
        }
        return buffer;
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "f.density" {
    try expectEqual(0, density(-inf, 3, 5));
    try expectEqual(0, density( inf, 3, 5));

    try expectEqual(inf, density(0, 1.8, 5));
    try expectEqual(  1, density(0, 2  , 5));
    try expectEqual(  0, density(0, 2.2, 5));

    try expectApproxEqRel(0                 , density(0, 3, 5), eps);
    try expectApproxEqRel(0.3611744789422851, density(1, 3, 5), eps);
    try expectApproxEqRel(0.1428963909075316, density(2, 3, 5), eps);
}

test "f.probability" {
    try expectEqual(0, probability(-inf, 3, 5));
    try expectEqual(1, probability( inf, 3, 5));

    try expectApproxEqRel(0                 , probability(0, 3, 5), eps);
    try expectApproxEqRel(0.5351452100063649, probability(1, 3, 5), eps);
    try expectApproxEqRel(0.7673760819999214, probability(2, 3, 5), eps);
}

test "f.quantile" {
    try expectApproxEqRel(0                 , quantile(0  , 3, 5), eps);
    try expectApproxEqRel(0.3372475270245997, quantile(0.2, 3, 5), eps);
    try expectApproxEqRel(0.6821342707772098, quantile(0.4, 3, 5), eps);
    try expectApproxEqRel(1.1978047828924259, quantile(0.6, 3, 5), eps);
    try expectApproxEqRel(2.2530173716474851, quantile(0.8, 3, 5), eps);
    try expectEqual      (inf               , quantile(1  , 3, 5)     );
}

test "f.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.73d1aa315be37p-1, random.single(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.bf5ec1a08f87bp-2, random.single(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.cbddabd676b5fp-1, random.single(gen, 3, 5), eps);
}
