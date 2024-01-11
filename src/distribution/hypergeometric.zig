//! Parameters:
//! - N: `N` ∈ {0,1,2,⋯}
//! - K: `K` ∈ {0,1,⋯,N}
//! - n: `n` ∈ {0,1,⋯,N}

const std = @import("std");
const math = @import("math.zig");
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

pub const parameters = 3;

/// p(x) = (K x) (N - K n - x) / (N n).
pub fn density(x: f64, N: u64, K: u64, n: u64) f64 {
    assert(K <= N and n <= N);
    assert(!isNan(x));
    const Nf = @as(f64, @floatFromInt(N));
    const Kf = @as(f64, @floatFromInt(K));
    const nf = @as(f64, @floatFromInt(n));
    if (x < 0 or x > nf or x > Kf or x != @round(x)) {
        return 0;
    }
    const num1 = math.lbinomial(Kf, x);
    const num2 = math.lbinomial(Nf - Kf, nf - x);
    const den = math.lbinomial(Nf, nf);
    return @exp(num1 + num2 - den);
}

/// No closed form.
pub fn probability(q: f64, N: u64, K: u64, n: u64) f64 {
    assert(K <= N and n <= N);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    const Nf = @as(f64, @floatFromInt(N));
    const Kf = @as(f64, @floatFromInt(K));
    const nf = @as(f64, @floatFromInt(n));
    if (q >= nf or q >= Kf) {
        return 1;
    }
    if (K == N or n == N) {
        return 0;
    }
    const mass_num = math.lbinomial(Nf - Kf, nf);
    const mass_den = math.lbinomial(Nf, nf);
    var mass = @exp(mass_num - mass_den);
    var cumu: f64 = mass;
    const qu = @as(usize, @intFromFloat(q));
    for (0..qu) |x| {
        const num = @as(f64, @floatFromInt((K - x) * (n - x)));
        const den = @as(f64, @floatFromInt((x + 1) * (N - K - n + x + 1)));
        mass *= num / den;
        cumu += mass;
    }
    return cumu;
}

/// No closed form.
pub fn quantile(p: f64, N: u64, K: u64, n: u64) f64 {
    assert(K <= N and n <= N);
    assert(0 <= p and p <= 1);
    if (n == 0 or K == 0) {
        return 0;
    }
    const Nf = @as(f64, @floatFromInt(N));
    const Kf = @as(f64, @floatFromInt(K));
    const nf = @as(f64, @floatFromInt(n));
    if (K == N) {
        return nf;
    }
    if (n == N) {
        return Kf;
    }
    if (p == 1) {
        return @min(nf, Kf);
    }
    const mass_num = math.lbinomial(Nf - Kf, nf);
    const mass_den = math.lbinomial(Nf, nf);
    var mass = @exp(mass_num - mass_den);
    var cumu = mass;
    var hyper: u64 = 0;
    while (p >= cumu) : (hyper += 1) {
        const num = @as(f64, @floatFromInt((K - hyper) * (n - hyper)));
        const den = @as(f64, @floatFromInt((hyper + 1) * (N - K - n + hyper + 1)));
        mass *= num / den;
        cumu += mass;
    }
    return @floatFromInt(hyper);
}

/// Uses the quantile function.
pub const random = struct {
    fn implementation(generator: std.rand.Random, N: u64, K: u64, n: u64) f64 {
        if (n == 0 or K == 0) {
            return 0;
        }
        const Nf = @as(f64, @floatFromInt(N));
        const Kf = @as(f64, @floatFromInt(K));
        const nf = @as(f64, @floatFromInt(n));
        if (K == N) {
            return nf;
        }
        if (n == N) {
            return Kf;
        }
        const mass_num = math.lbinomial(Nf - Kf, nf);
        const mass_den = math.lbinomial(Nf, nf);
        var mass = @exp(mass_num - mass_den);
        var cumu = mass;
        var hyper: u64 = 0;
        const uni = generator.float(f64);
        while (uni >= cumu) : (hyper += 1) {
            const num = @as(f64, @floatFromInt((K - hyper) * (n - hyper)));
            const den = @as(f64, @floatFromInt((hyper + 1) * (N - K - n + hyper + 1)));
            mass *= num / den;
            cumu += mass;
        }
        return @floatFromInt(hyper);
    }

    pub fn single(generator: std.rand.Random, N: u64, K: u64, n: u64) f64 {
        assert(K <= N and n <= N);
        return implementation(generator, N, K, n);
    }

    pub fn buffer(buf: []f64, generator: std.rand.Random, N: u64, K: u64, n: u64) []f64 {
        assert(K <= N and n <= N);
        for (buf) |*x| {
            x.* = implementation(generator, N, K, n);
        }
        return buf;
    }

    pub fn alloc(allocator: std.mem.Allocator, generator: std.rand.Random, nn: usize, N: u64, K: u64, n: u64) ![]f64 {
        const slice = try allocator.alloc(f64, nn);
        return buffer(slice, generator, N, K, n);
    }
};

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

test "hypergeometric.density" {
    try expectEqual(0, density(-inf, 10, 2, 5));
    try expectEqual(0, density( inf, 10, 2, 5));

    try expectEqual(1, density( 0, 10,  2,  0));
    try expectEqual(0, density( 1, 10,  2,  0));
    try expectEqual(1, density( 0, 10,  0,  5));
    try expectEqual(0, density( 1, 10,  0,  5));
    try expectEqual(0, density( 4, 10, 10,  5));
    try expectEqual(1, density( 5, 10, 10,  5));
    try expectEqual(0, density( 6, 10, 10,  5));
    try expectEqual(0, density( 1, 10,  2, 10));
    try expectEqual(1, density( 2, 10,  2, 10));
    try expectEqual(0, density( 3, 10,  2, 10));
    try expectEqual(0, density( 9, 10, 10, 10));
    try expectEqual(1, density(10, 10, 10, 10));
    try expectEqual(0, density(11, 10, 10, 10));

    try expectApproxEqRel(0                 , density(-0.1, 10, 2, 5), eps);
    try expectApproxEqRel(0.2222222222222222, density( 0  , 10, 2, 5), eps);
    try expectApproxEqRel(0                 , density( 0.1, 10, 2, 5), eps);
    try expectApproxEqRel(0                 , density( 0.9, 10, 2, 5), eps);
    try expectApproxEqRel(0.5555555555555556, density( 1  , 10, 2, 5), eps);
    try expectApproxEqRel(0                 , density( 1.1, 10, 2, 5), eps);
}

test "hypergeometric.probability" {
    try expectEqual(0, probability(-inf, 10, 2, 5));
    try expectEqual(1, probability( inf, 10, 2, 5));

    try expectEqual(1, probability( 0, 10,  2, 0 ));
    try expectEqual(1, probability( 1, 10,  2, 0 ));
    try expectEqual(1, probability( 0, 10,  0, 5 ));
    try expectEqual(1, probability( 1, 10,  0, 5 ));
    try expectEqual(0, probability( 4, 10, 10, 5 ));
    try expectEqual(1, probability( 5, 10, 10, 5 ));
    try expectEqual(1, probability( 6, 10, 10, 5 ));
    try expectEqual(0, probability( 1, 10,  2, 10));
    try expectEqual(1, probability( 2, 10,  2, 10));
    try expectEqual(1, probability( 3, 10,  2, 10));
    try expectEqual(0, probability( 9, 10, 10, 10));
    try expectEqual(1, probability(10, 10, 10, 10));
    try expectEqual(1, probability(11, 10, 10, 10));

    try expectApproxEqRel(0                 , probability(-0.1, 10, 2, 5), eps);
    try expectApproxEqRel(0.2222222222222222, probability( 0  , 10, 2, 5), eps);
    try expectApproxEqRel(0.2222222222222222, probability( 0.1, 10, 2, 5), eps);
    try expectApproxEqRel(0.2222222222222222, probability( 0.9, 10, 2, 5), eps);
    try expectApproxEqRel(0.7777777777777778, probability( 1  , 10, 2, 5), eps);
    try expectApproxEqRel(0.7777777777777778, probability( 1.1, 10, 2, 5), eps);
}

test "hypergeometric.quantile" {
    try expectEqual( 0, quantile(0  , 10,  2, 0 ));
    try expectEqual( 0, quantile(0.5, 10,  2, 0 ));
    try expectEqual( 0, quantile(1  , 10,  2, 0 ));
    try expectEqual( 0, quantile(0  , 10,  0, 5 ));
    try expectEqual( 0, quantile(0.5, 10,  0, 5 ));
    try expectEqual( 0, quantile(1  , 10,  0, 5 ));
    try expectEqual( 5, quantile(0  , 10, 10, 5 ));
    try expectEqual( 5, quantile(0.5, 10, 10, 5 ));
    try expectEqual( 5, quantile(1  , 10, 10, 5 ));
    try expectEqual( 2, quantile(0  , 10,  2, 10));
    try expectEqual( 2, quantile(0.5, 10,  2, 10));
    try expectEqual( 2, quantile(1  , 10,  2, 10));
    try expectEqual(10, quantile(0  , 10, 10, 10));
    try expectEqual(10, quantile(0.5, 10, 10, 10));
    try expectEqual(10, quantile(1  , 10, 10, 10));

    try expectEqual(0, quantile(0                 , 10, 2, 5));
    try expectEqual(0, quantile(0.2222222222222221, 10, 2, 5));
    try expectEqual(0, quantile(0.2222222222222222, 10, 2, 5));
    try expectEqual(1, quantile(0.2222222222222223, 10, 2, 5));
    try expectEqual(1, quantile(0.7777777777777777, 10, 2, 5));
    try expectEqual(1, quantile(0.7777777777777778, 10, 2, 5));
    try expectEqual(2, quantile(0.7777777777777780, 10, 2, 5));
    try expectEqual(2, quantile(1                 , 10, 2, 5));
}

test "hypergeometric.random" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual( 0, random.implementation(gen, 10,  2, 0 ));
    try expectEqual( 0, random.implementation(gen, 10,  2, 0 ));
    try expectEqual( 0, random.implementation(gen, 10,  2, 0 ));
    try expectEqual( 0, random.implementation(gen, 10,  0, 5 ));
    try expectEqual( 0, random.implementation(gen, 10,  0, 5 ));
    try expectEqual( 0, random.implementation(gen, 10,  0, 5 ));
    try expectEqual( 5, random.implementation(gen, 10, 10, 5 ));
    try expectEqual( 5, random.implementation(gen, 10, 10, 5 ));
    try expectEqual( 5, random.implementation(gen, 10, 10, 5 ));
    try expectEqual( 2, random.implementation(gen, 10,  2, 10));
    try expectEqual( 2, random.implementation(gen, 10,  2, 10));
    try expectEqual( 2, random.implementation(gen, 10,  2, 10));
    try expectEqual(10, random.implementation(gen, 10, 10, 10));
    try expectEqual(10, random.implementation(gen, 10, 10, 10));
    try expectEqual(10, random.implementation(gen, 10, 10, 10));

    try expectEqual(1, random.implementation(gen, 10, 2, 5));
    try expectEqual(1, random.implementation(gen, 10, 2, 5));
    try expectEqual(1, random.implementation(gen, 10, 2, 5));
}
