//! Support: {0,1,2,⋯}
//!
//! Parameters:
//! - n: `size` ∈ {1,2,⋯}
//! - p: `prob` ∈ (0,1]

const std = @import("std");
const gamma = @import("gamma.zig");
const poisson = @import("poisson.zig");
const special = @import("../special.zig");
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// p(x) = (x + n - 1 x) p^n (1 - p)^x
pub fn density(x: f64, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    assert(!isNan(x));
    if (x < 0 or x == inf or x != @round(x)) {
        return 0;
    }
    if (prob == 1) {
        return if (x == 0) 1 else 0;
    }
    const n: f64 = @floatFromInt(size);
    const binom = special.lbinomial(n + x - 1, x);
    const log = binom + n * @log(prob) + x * std.math.log1p(-prob);
    return @exp(log);
}

/// No closed form
pub fn probability(q: f64, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    if (q == inf or prob == 1) {
        return 1;
    }
    const n: f64 = @floatFromInt(size);
    return special.beta_probability(n, @floor(q) + 1, prob);
}

/// No closed form
pub fn quantile(p: f64, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    assert(0 <= p and p <= 1);
    if (p == 0 or prob == 1) {
        return 0;
    }
    if (p == 1) {
        return inf;
    }
    const n: f64 = @floatFromInt(size);
    const mean = n * (1 - prob) / prob;
    if (mean < 250) {
        const initial_mass = std.math.pow(f64, prob, n);
        return linearSearch(p, n, 1 - prob, initial_mass);
    }
    const initial_nbin = @ceil(mean);
    const initial_mass = density(initial_nbin, size, prob);
    const initial_cumu = probability(initial_nbin, size, prob);
    return guidedSearch(p, n, 1 - prob, initial_nbin, initial_mass, initial_cumu);
}

pub fn random(generator: std.Random, size: u64, prob: f64) f64 {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    if (prob == 1) {
        return 0;
    }
    const n: f64 = @floatFromInt(size);
    const qrob = 1 - prob;
    const pq = prob / qrob;
    const mean = n / pq;
    if (mean < 150) {
        const initial_mass = std.math.pow(f64, prob, n);
        const uni = generator.float(f64);
        return linearSearch(uni, n, qrob, initial_mass);
    }
    const lambda = gamma.random(generator, n, pq);
    return poisson.random(generator, lambda);
}

pub fn fill(buffer: []f64, generator: std.Random, size: u64, prob: f64) void {
    assert(0 < prob and prob <= 1);
    assert(size != 0);
    if (prob == 1) {
        return @memset(buffer, 0);
    }
    const n: f64 = @floatFromInt(size);
    const qrob = 1 - prob;
    const pq = prob / qrob;
    const mean = n / pq;
    if (buffer.len < 15 and mean < 15) {
        const initial_mass = std.math.pow(f64, prob, n);
        for (buffer) |*x| {
            const uni = generator.float(f64);
            x.* = linearSearch(uni, n, qrob, initial_mass);
        }
        return;
    }
    if (mean < 15000) {
        const initial_nbin = @ceil(mean);
        const initial_mass = density(initial_nbin, size, prob);
        const initial_cumu = probability(initial_nbin, size, prob);
        for (buffer) |*x| {
            const uni = generator.float(f64);
            x.* = guidedSearch(uni, n, qrob, initial_nbin, initial_mass, initial_cumu);
        }
        return;
    }
    for (buffer) |*x| {
        const lambda = gamma.random(generator, n, pq);
        x.* = poisson.random(generator, lambda);
    }
}

fn linearSearch(p: f64, n: f64, q: f64, initial_mass: f64) f64 {
    var nbin: f64 = 0;
    var mass = initial_mass;
    var cumu = mass;
    while (cumu <= p) {
        const num = q * (n + nbin);
        nbin += 1;
        mass *= num / nbin;
        cumu += mass;
    }
    return nbin;
}

fn guidedSearch(p: f64, n: f64, q: f64, initial_nbin: f64, initial_mass: f64, initial_cumu: f64) f64 {
    var nbin = initial_nbin;
    var mass = initial_mass;
    var cumu = initial_cumu;
    if (initial_cumu <= p) {
        while (cumu <= p) {
            const num = q * (n + nbin);
            nbin += 1;
            mass *= num / nbin;
            cumu += mass;
        }
    } else {
        while (true) {
            cumu -= mass;
            if (cumu <= p) {
                break;
            }
            const num = nbin;
            nbin -= 1;
            mass *= num / (q * (n + nbin));
        }
    }
    return nbin;
}

export fn rv_negative_binomial_density(x: f64, size: u64, prob: f64) f64 {
    return density(x, size, prob);
}
export fn rv_negative_binomial_probability(q: f64, size: u64, prob: f64) f64 {
    return probability(q, size, prob);
}
export fn rv_negative_binomial_quantile(p: f64, size: u64, prob: f64) f64 {
    return quantile(p, size, prob);
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 10, 0.2));
    try expectEqual(0, density( inf, 10, 0.2));

    try expectEqual(1, density(0, 10, 1));
    try expectEqual(0, density(1, 10, 1));

    try expectApproxEqRel(0           , density(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.0000001024, density( 0  , 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(0.0000008192, density( 1  , 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 1.1, 10, 0.2), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 10, 0.2));
    try expectEqual(1, probability( inf, 10, 0.2));

    try expectEqual(1, probability(0, 10, 1));
    try expectEqual(1, probability(1, 10, 1));

    try expectApproxEqRel(0           , probability(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.0000001024, probability( 0  , 10, 0.2), eps);
    try expectApproxEqRel(0.0000001024, probability( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.0000001024, probability( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(0.0000009216, probability( 1  , 10, 0.2), eps);
    try expectApproxEqRel(0.0000009216, probability( 1.1, 10, 0.2), eps);
}

test quantile {
    try expectEqual(0, quantile(0  , 10, 1));
    try expectEqual(0, quantile(0.5, 10, 1));
    try expectEqual(0, quantile(1  , 10, 1));

    try expectEqual(  0, quantile(0           , 10, 0.2));
    try expectEqual(  0, quantile(0.0000001023, 10, 0.2));
    try expectEqual(  0, quantile(0.0000001024, 10, 0.2));
    try expectEqual(  1, quantile(0.0000001025, 10, 0.2));
    try expectEqual(  1, quantile(0.0000009215, 10, 0.2));
    try expectEqual(  1, quantile(0.0000009216, 10, 0.2));
    try expectEqual(  2, quantile(0.0000009217, 10, 0.2));
    try expectEqual(inf, quantile(1           , 10, 0.2));
}
