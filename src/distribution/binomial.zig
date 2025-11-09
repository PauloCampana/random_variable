//! Support: {0,1,⋯,n}
//!
//! Parameters:
//! - n: `size` ∈ {0,1,2,⋯}
//! - p: `prob` ∈ [0,1]

const std = @import("std");
const special = @import("../special.zig");
const assert = std.debug.assert;
const isNan = std.math.isNan;
const inf = std.math.inf(f64);

/// p(x) = (n x) p^x (1 - p)^(n - x)
pub fn density(x: f64, size: u64, prob: f64) callconv(.c) f64 {
    assert(0 <= prob and prob <= 1);
    assert(!isNan(x));
    const n: f64 = @floatFromInt(size);
    if (x < 0 or x > n or x != @round(x)) {
        return 0;
    }
    if (prob == 0) {
        return if (x == 0) 1 else 0;
    }
    if (prob == 1) {
        return if (x == n) 1 else 0;
    }
    const binom = special.lbinomial(n, x);
    const log = binom + x * @log(prob) + (n - x) * std.math.log1p(-prob);
    return @exp(log);
}

/// No closed form
pub fn probability(q: f64, size: u64, prob: f64) callconv(.c) f64 {
    assert(0 <= prob and prob <= 1);
    assert(!isNan(q));
    const n: f64 = @floatFromInt(size);
    if (q < 0) {
        return 0;
    }
    if (q >= n) {
        return 1;
    }
    const fq = @floor(q);
    return special.beta.probability(1 - prob, n - fq, fq + 1);
}

/// No closed form
pub fn survival(t: f64, size: u64, prob: f64) callconv(.c) f64 {
    assert(0 <= prob and prob <= 1);
    assert(!isNan(t));
    const n: f64 = @floatFromInt(size);
    if (t < 0) {
        return 1;
    }
    if (t >= n) {
        return 0;
    }
    const ft = @floor(t);
    return special.beta.probability(prob, ft + 1, n - ft);
}

/// No closed form
pub fn quantile(p: f64, size: u64, prob: f64) callconv(.c) f64 {
    assert(0 <= prob and prob <= 1);
    assert(0 <= p and p <= 1);
    const n: f64 = @floatFromInt(size);
    if (p == 0) {
        return 0;
    }
    if (p == 1) {
        return n;
    }
    const qrob = 1 - prob;
    const mean = n * prob;
    if (prob == 0 or size == 0) {
        return 0;
    }
    if (prob == 1) {
        return n;
    }
    if (mean < 500) {
        if (prob < 0.5) {
            const initial_mass = std.math.pow(f64, qrob, n);
            return linearSearch(p, n, prob / qrob, initial_mass);
        } else {
            const initial_mass = std.math.pow(f64, prob, n);
            return n - linearSearch(p, n, qrob / prob, initial_mass);
        }
    }
    const initial_bino = @ceil(mean);
    const initial_mass = density(initial_bino, size, prob);
    const initial_cumu = probability(initial_bino, size, prob);
    return guidedSearch(p, n, prob / qrob, initial_bino, initial_mass, initial_cumu);
}

pub fn random(generator: std.Random, size: u64, prob: f64) f64 {
    assert(0 <= prob and prob <= 1);
    const n: f64 = @floatFromInt(size);
    const qrob = 1 - prob;
    const mean = n * prob;
    if (prob == 0 or size == 0) {
        return 0;
    }
    if (prob == 1) {
        return n;
    }
    if (prob == 0.5) {
        return popCount(generator, size);
    }
    if (mean < 1000) {
        if (prob < 0.5) {
            const initial_mass = std.math.pow(f64, qrob, n);
            const uni = generator.float(f64);
            return linearSearch(uni, n, prob / qrob, initial_mass);
        } else {
            const initial_mass = std.math.pow(f64, prob, n);
            const uni = generator.float(f64);
            return n - linearSearch(uni, n, qrob / prob, initial_mass);
        }
    }
    const initial_bino = @ceil(mean);
    const initial_mass = density(initial_bino, size, prob);
    const initial_cumu = probability(initial_bino, size, prob);
    const uni = generator.float(f64);
    return guidedSearch(uni, n, prob / qrob, initial_bino, initial_mass, initial_cumu);
}

pub fn fill(buffer: []f64, generator: std.Random, size: u64, prob: f64) void {
    assert(0 <= prob and prob <= 1);
    const n: f64 = @floatFromInt(size);
    const qrob = 1 - prob;
    const mean = n * prob;
    if (prob == 0 or size == 0) {
        return @memset(buffer, 0);
    }
    if (prob == 1) {
        return @memset(buffer, n);
    }
    if (prob == 0.5) {
        for (buffer) |*x| {
            x.* = popCount(generator, size);
        }
        return;
    }
    if (buffer.len < 20) {
        if (prob < 0.5) {
            const pq = prob / qrob;
            const initial_mass = std.math.pow(f64, qrob, n);
            for (buffer) |*x| {
                const uni = generator.float(f64);
                x.* = linearSearch(uni, n, pq, initial_mass);
            }
        } else {
            const qp = qrob / prob;
            const initial_mass = std.math.pow(f64, prob, n);
            for (buffer) |*x| {
                const uni = generator.float(f64);
                x.* = n - linearSearch(uni, n, qp, initial_mass);
            }
        }
        return;
    }
    const pq = prob / qrob;
    const initial_bino = @ceil(mean);
    const initial_mass = density(initial_bino, size, prob);
    const initial_cumu = probability(initial_bino, size, prob);
    for (buffer) |*x| {
        const uni = generator.float(f64);
        x.* = guidedSearch(uni, n, pq, initial_bino, initial_mass, initial_cumu);
    }
}

fn linearSearch(p: f64, n: f64, pq: f64, initial_mass: f64) f64 {
    var mass = initial_mass;
    var cumu = mass;
    var bin: f64 = 0;
    while (cumu <= p) {
        const num = n - bin;
        bin += 1;
        mass *= pq * num / bin;
        cumu += mass;
    }
    return bin;
}

fn guidedSearch(p: f64, n: f64, pq: f64, initial_bino: f64, initial_mass: f64, initial_cumu: f64) f64 {
    var bino = initial_bino;
    var mass = initial_mass;
    var cumu = initial_cumu;
    if (initial_cumu <= p) {
        while (cumu <= p) {
            const num = n - bino;
            bino += 1;
            mass *= pq * num / bino;
            cumu += mass;
        }
    } else {
        while (true) {
            cumu -= mass;
            if (cumu <= p) {
                break;
            }
            const num = bino;
            bino -= 1;
            mass *= num / (pq * (n - bino));
        }
    }
    return bino;
}

fn popCount(generator: std.Random, size: u64) f64 {
    const remainder = size % 64;
    var bino: u64 = 0;
    var i: u64 = 0;

    if (remainder != 0) {
        const uni64 = generator.int(u64);
        const unisize = uni64 >> @truncate(64 - remainder);
        bino += @popCount(unisize);
        i += remainder;
    }
    while (i < size) : (i += 64) {
        const uni64 = generator.int(u64);
        bino += @popCount(uni64);
    }
    return @floatFromInt(bino);
}

comptime {
    @export(&density, .{ .name = "rv_binomial_density" });
    @export(&probability, .{ .name = "rv_binomial_probability" });
    @export(&survival, .{ .name = "rv_binomial_survival" });
    @export(&quantile, .{ .name = "rv_binomial_quantile" });
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test density {
    try expectEqual(0, density(-inf, 10, 0.2));
    try expectEqual(0, density( inf, 10, 0.2));

    try expectEqual(1, density( 0,  0, 0.2));
    try expectEqual(0, density( 1,  0, 0.2));
    try expectEqual(1, density( 0, 10, 0  ));
    try expectEqual(0, density( 1, 10, 0  ));
    try expectEqual(0, density( 9, 10, 1  ));
    try expectEqual(1, density(10, 10, 1  ));
    try expectEqual(0, density(11, 10, 1  ));

    try expectApproxEqRel(0           , density(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.1073741824, density( 0  , 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(0.2684354560, density( 1  , 10, 0.2), eps);
    try expectApproxEqRel(0           , density( 1.1, 10, 0.2), eps);
}

test probability {
    try expectEqual(0, probability(-inf, 10, 0.2));
    try expectEqual(1, probability( inf, 10, 0.2));

    try expectEqual(1, probability( 0,  0, 0.2));
    try expectEqual(1, probability( 1,  0, 0.2));
    try expectEqual(1, probability( 0, 10, 0  ));
    try expectEqual(1, probability( 1, 10, 0  ));
    try expectEqual(0, probability( 9, 10, 1  ));
    try expectEqual(1, probability(10, 10, 1  ));
    try expectEqual(1, probability(11, 10, 1  ));

    try expectApproxEqRel(0           , probability(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.1073741824, probability( 0  , 10, 0.2), eps);
    try expectApproxEqRel(0.1073741824, probability( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.1073741824, probability( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(0.3758096384, probability( 1  , 10, 0.2), eps);
    try expectApproxEqRel(0.3758096384, probability( 1.1, 10, 0.2), eps);
}

test survival {
    try expectEqual(1, survival(-inf, 10, 0.2));
    try expectEqual(0, survival( inf, 10, 0.2));

    try expectEqual(0, survival( 0,  0, 0.2));
    try expectEqual(0, survival( 1,  0, 0.2));
    try expectEqual(0, survival( 0, 10, 0  ));
    try expectEqual(0, survival( 1, 10, 0  ));
    try expectEqual(1, survival( 9, 10, 1  ));
    try expectEqual(0, survival(10, 10, 1  ));
    try expectEqual(0, survival(11, 10, 1  ));

    try expectApproxEqRel(1           , survival(-0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.8926258176, survival( 0  , 10, 0.2), eps);
    try expectApproxEqRel(0.8926258176, survival( 0.1, 10, 0.2), eps);
    try expectApproxEqRel(0.8926258176, survival( 0.9, 10, 0.2), eps);
    try expectApproxEqRel(0.6241903616, survival( 1  , 10, 0.2), eps);
    try expectApproxEqRel(0.6241903616, survival( 1.1, 10, 0.2), eps);
}

test quantile {
    try expectEqual( 0, quantile(0  ,  0, 0.2));
    try expectEqual( 0, quantile(0.5,  0, 0.2));
    try expectEqual( 0, quantile(1  ,  0, 0.2));
    try expectEqual( 0, quantile(0  , 10, 0  ));
    try expectEqual( 0, quantile(0.5, 10, 0  ));
    try expectEqual(10, quantile(1  , 10, 0  ));
    try expectEqual( 0, quantile(0  , 10, 1  ));
    try expectEqual(10, quantile(0.5, 10, 1  ));
    try expectEqual(10, quantile(1  , 10, 1  ));

    try expectEqual( 0, quantile(0           , 10, 0.2));
    try expectEqual( 0, quantile(0.1073741823, 10, 0.2));
    try expectEqual( 0, quantile(0.1073741824, 10, 0.2));
    try expectEqual( 1, quantile(0.1073741825, 10, 0.2));
    try expectEqual( 1, quantile(0.3758096383, 10, 0.2));
    try expectEqual( 1, quantile(0.3758096384, 10, 0.2));
    try expectEqual( 2, quantile(0.3758096385, 10, 0.2));
    try expectEqual(10, quantile(1           , 10, 0.2));
}
