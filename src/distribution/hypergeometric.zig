//! Support: {max(0, n + K - N),1,⋯,min(n, K)}
//!
//! Parameters:
//! - N: `N` ∈ {0,1,2,⋯}
//! - K: `K` ∈ {0,1,⋯,N}
//! - n: `n` ∈ {0,1,⋯,N}

const std = @import("std");
const special = @import("../special.zig");
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// p(x) = (K x) (N - K n - x) / (N n)
pub fn density(x: f64, N: u64, K: u64, n: u64) f64 {
    assert(K <= N and n <= N);
    assert(!isNan(x));
    const Nf: f64 = @floatFromInt(N);
    const Kf: f64 = @floatFromInt(K);
    const nf: f64 = @floatFromInt(n);
    if (x < 0 or x > nf or x > Kf or x != @round(x)) {
        return 0;
    }
    const num1 = special.lbinomial(Kf, x);
    const num2 = special.lbinomial(Nf - Kf, nf - x);
    const den = special.lbinomial(Nf, nf);
    return @exp(num1 + num2 - den);
}

/// No closed form
pub fn probability(q: f64, N: u64, K: u64, n: u64) f64 {
    assert(K <= N and n <= N);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    const Kf: f64 = @floatFromInt(K);
    const nf: f64 = @floatFromInt(n);
    if (q >= nf or q >= Kf) {
        return 1;
    }
    if (K == N or n == N) {
        return 0;
    }
    var hypr = if (n + K < N) 0 else n + K - N;
    var mass = density(@floatFromInt(hypr), N, K, n);
    var cumu = mass;
    for (0..@intFromFloat(q)) |_| {
        const num: f64 = @floatFromInt((K - hypr) * (n - hypr));
        hypr += 1;
        const den: f64 = @floatFromInt(hypr * (hypr + N - K - n));
        mass *= num / den;
        cumu += mass;
    }
    return cumu;
}

/// No closed form
pub fn quantile(p: f64, N: u64, K: u64, n: u64) f64 {
    assert(K <= N and n <= N);
    assert(0 <= p and p <= 1);
    if (p == 0) {
        return if (n + K < N) 0 else @floatFromInt(n + K - N);
    }
    if (p == 1) {
        return @floatFromInt(@min(n, K));
    }
    if (n == 0 or K == 0) {
        return 0;
    }
    if (K == N) {
        return @floatFromInt(n);
    }
    if (n == N) {
        return @floatFromInt(K);
    }
    const initial = if (n + K < N) 0 else n + K - N;
    const initial_mass = density(@floatFromInt(initial), N, K, n);
    return linearSearch(p, N, K, n, initial, initial_mass);
}

pub fn random(generator: std.Random, N: u64, K: u64, n: u64) f64 {
    assert(K <= N and n <= N);
    if (n == 0 or K == 0) {
        return 0;
    }
    if (K == N) {
        return @floatFromInt(n);
    }
    if (n == N) {
        return @floatFromInt(K);
    }
    const initial = if (n + K < N) 0 else n + K - N;
    const initial_mass = density(@floatFromInt(initial), N, K, n);
    const uni = generator.float(f64);
    return linearSearch(uni, N, K, n, initial, initial_mass);
}

pub fn fill(buffer: []f64, generator: std.Random, N: u64, K: u64, n: u64) []f64 {
    assert(K <= N and n <= N);
    if (n == 0 or K == 0) {
        @memset(buffer, 0);
        return buffer;
    }
    if (K == N) {
        @memset(buffer, @floatFromInt(n));
        return buffer;
    }
    if (n == N) {
        @memset(buffer, @floatFromInt(K));
        return buffer;
    }
    const initial = if (n + K < N) 0 else n + K - N;
    const initial_mass = density(@floatFromInt(initial), N, K, n);
    for (buffer) |*x| {
        const uni = generator.float(f64);
        x.* = linearSearch(uni, N, K, n, initial, initial_mass);
    }
    return buffer;
}

fn linearSearch(p: f64, N: u64, K: u64, n: u64, initial: u64, initial_mass: f64) f64 {
    var hypr = initial;
    var mass = initial_mass;
    var cumu = mass;
    while (cumu <= p) {
        const num: f64 = @floatFromInt((K - hypr) * (n - hypr));
        hypr += 1;
        const den: f64 = @floatFromInt(hypr * (hypr + N - K - n));
        mass *= num / den;
        cumu += mass;
    }
    return @floatFromInt(hypr);
}

export fn rv_hypergeometric_density(x: f64, N: u64, K: u64, n: u64) f64 {
    return density(x, N, K, n);
}
export fn rv_hypergeometric_probability(q: f64, N: u64, K: u64, n: u64) f64 {
    return probability(q, N, K, n);
}
export fn rv_hypergeometric_quantile(p: f64, N: u64, K: u64, n: u64) f64 {
    return quantile(p, N, K, n);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
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

test probability {
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

test quantile {
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
