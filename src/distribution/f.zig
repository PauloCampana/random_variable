//! Parameters:
//! - n: `df1` ∈ (0,∞)
//! - m: `df2` ∈ (0,∞)

const std = @import("std");
const gamma = @import("gamma.zig");
const lgamma = @import("../thirdyparty/prob.zig").lnGamma;
const incompleteBeta = @import("../thirdyparty/prob.zig").incompleteBeta;
const inverseIncompleteBeta = @import("../thirdyparty/prob.zig").inverseIncompleteBeta;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

pub const parameters = 2;
pub const support = [2]f64 {0, inf};

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
    const df3 = df1 / 2;
    const df4 = df2 / 2;
    const df5 = df3 + df4;
    const num1 = df3 * @log(df1) + df4 * @log(df2) + (df3 - 1) * @log(x);
    const num2 = -df5 * @log(df2 + df1 * x);
    const den = lgamma(df3) + lgamma(df4) - lgamma(df5);
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
    fn implementation(generator: std.rand.Random, df1: f64, df2: f64) f64 {
        const chinum = gamma.random.implementation(generator, 0.5 * df1, 1);
        const chiden = gamma.random.implementation(generator, 0.5 * df2, 1);
        return chinum / chiden * df2 / df1;
    }

    pub fn single(generator: std.rand.Random, df1: f64, df2: f64) f64 {
        assert(isFinite(df1) and isFinite(df2));
        assert(df1 > 0 and df2 > 0);
        return implementation(generator, df1, df2);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, df1: f64, df2: f64) []f64 {
        assert(isFinite(df1) and isFinite(df2));
        assert(df1 > 0 and df2 > 0);
        for (buf) |*x| {
            x.* = implementation(generator, df1, df2);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, df1: f64, df2: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, df1, df2);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

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

test "f.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(0x1.73d1aa315be37p-1, random.implementation(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.bf5ec1a08f87bp-2, random.implementation(gen, 3, 5), eps);
    try expectApproxEqRel(0x1.cbddabd676b5fp-1, random.implementation(gen, 3, 5), eps);
}
