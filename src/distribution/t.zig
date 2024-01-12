//! Parameters:
//! - ν: `df` ∈ (0,∞)

const std = @import("std");
const gamma = @import("gamma.zig");
const math = @import("../math.zig");
const incompleteBeta = @import("../thirdyparty/prob.zig").incompleteBeta;
const inverseIncompleteBeta = @import("../thirdyparty/prob.zig").inverseIncompleteBeta;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const parameters = 1;
pub const support = [2]f64 {-inf, inf};

/// f(x) (ν / (ν + x^2))^((ν + 1) / 2) / (sqrt(ν) beta(ν / 2, 1 / 2)).
pub fn density(x: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(!isNan(x));
    const num = (0.5 * df + 0.5) * @log(df / (df + x * x));
    const den = 0.5 * @log(df) + math.lbeta(0.5 * df, 0.5);
    return @exp(num - den);
}

/// No closed form.
pub fn probability(q: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(!isNan(q));
    if (std.math.isInf(q)) {
        return if (q < 0) 0 else 1;
    }
    const z = q * q;
    if (q < 0) {
        const p = df / (df + z);
        return 0.5 * incompleteBeta(0.5 * df, 0.5, p);
    } else {
        const p = z / (df + z);
        return 0.5 * incompleteBeta(0.5, 0.5 * df, p) + 0.5;
    }
}

/// No closed form.
pub fn quantile(p: f64, df: f64) f64 {
    assert(isFinite(df));
    assert(df > 0);
    assert(0 <= p and p <= 1);
    if (p < 0.5) {
        const q = inverseIncompleteBeta(0.5 * df, 0.5, 2 * p);
        return -@sqrt(df / q - df);
    } else {
        const q = inverseIncompleteBeta(0.5 * df, 0.5, 2 - 2 * p);
        return @sqrt(df / q - df);
    }
}

/// Uses the relation to Normal and Gamma distributions.
pub const random = struct {
    fn implementation(generator: std.rand.Random, df: f64) f64 {
        if (df == 1) {
            const uni = generator.float(f64);
            return @tan(std.math.pi * uni);
        }
        const nor = generator.floatNorm(f64);
        const chi = gamma.random.implementation(generator, 0.5 * df, 0.5);
        return nor * @sqrt(df / chi);
    }

    pub fn single(generator: std.rand.Random, df: f64) f64 {
        assert(isFinite(df));
        assert(df > 0);
        return implementation(generator, df);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, df: f64) []f64 {
        assert(isFinite(df));
        assert(df > 0);
        for (buf) |*x| {
            x.* = implementation(generator, df);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, n: usize, df: f64) ![]f64 {
        const slice = try allocator.alloc(f64, n);
        return buffer(slice, generator, df);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "t.density" {
    try expectEqual(0, density(-inf, 3));
    try expectEqual(0, density( inf, 3));

    try expectApproxEqRel(0.3675525969478613, density(0, 3), eps);
    try expectApproxEqRel(0.2067483357831720, density(1, 3), eps);
    try expectApproxEqRel(0.0675096606638929, density(2, 3), eps);
}

test "t.probability" {
    try expectEqual(0, probability(-inf, 3));
    try expectEqual(1, probability( inf, 3));

    try expectApproxEqRel(0.5               , probability(0, 3), eps);
    try expectApproxEqRel(0.8044988905221148, probability(1, 3), eps);
    try expectApproxEqRel(0.9303370157205784, probability(2, 3), eps);
}

test "t.quantile" {
    try expectEqual      (-inf               , quantile(0  , 3)     );
    try expectApproxEqRel(-0.9784723123633045, quantile(0.2, 3), eps);
    try expectApproxEqRel(-0.2766706623326898, quantile(0.4, 3), eps);
    try expectApproxEqRel( 0.2766706623326902, quantile(0.6, 3), eps);
    try expectApproxEqRel( 0.9784723123633039, quantile(0.8, 3), eps);
    try expectEqual      ( inf               , quantile(1  , 3)     );
}

test "t.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(-0x1.ed977ce651337p-2, random.implementation(gen, 3), eps);
    try expectApproxEqRel(-0x1.62bae37cf8d83p+1, random.implementation(gen, 3), eps);
    try expectApproxEqRel( 0x1.a5797fad46fcap-1, random.implementation(gen, 3), eps);
}
