//! Support: X ∈ {0,1,2,⋯}
//!
//! Parameters:
//! - λ: `lambda` ∈ (0,∞)

const std = @import("std");
const incompleteGamma = @import("../thirdyparty/prob.zig").incompleteGamma;
const assert = std.debug.assert;
const isFinite = std.math.isFinite;
const isNan = std.math.isNan;
const isInf = std.math.isInf;
const inf = std.math.inf(f64);

pub const discrete = true;
pub const parameters = 1;

/// p(x) = λ^x exp(-λ) / x!.
pub fn density(x: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(!isNan(x));
    if (x < 0 or isInf(x) or x != @round(x)) {
        return 0;
    }
    const log = -lambda + x * @log(lambda) - std.math.lgamma(f64, x + 1);
    return @exp(log);
}

/// No closed form.
pub fn probability(q: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(!isNan(q));
    if (q < 0) {
        return 0;
    }
    if (isInf(q)) {
        return 1;
    }
    return 1 - incompleteGamma(@floor(q) + 1, lambda);
}

/// No closed form.
pub fn quantile(p: f64, lambda: f64) f64 {
    assert(isFinite(lambda));
    assert(lambda > 0);
    assert(0 <= p and p <= 1);
    if (p == 1) {
        return inf;
    }
    if (lambda < 350) {
        return linearSearch(p, lambda);
    } else {
        const initial_pois = @ceil(lambda);
        const initial_mass = density(initial_pois, lambda);
        const initial_cumu = probability(initial_pois, lambda);
        return guidedSearch(p, lambda, initial_pois, initial_mass, initial_cumu);
    }
}

/// Uses the quantile function or rejection sampling for higher λ.
pub const random = struct {
    pub fn single(generator: std.rand.Random, lambda: f64) f64 {
        assert(isFinite(lambda));
        assert(lambda > 0);
        if (lambda < 130) {
            const uni = generator.float(f64);
            return linearSearch(uni, lambda);
        } else {
            const beta = std.math.pi / @sqrt(3 * lambda);
            const c = 0.801 - 0.687 / @sqrt(lambda);
            const k = @log(c) - lambda - @log(beta);
            return rejection(generator, lambda, beta, k);
        }
    }

    pub fn fill(buffer: []f64, generator: std.rand.Random, lambda: f64) []f64 {
        assert(isFinite(lambda));
        assert(lambda > 0);
        if (buffer.len < 15 and lambda < 15) {
            for (buffer) |*x| {
                const uni = generator.float(f64);
                x.* = linearSearch(uni, lambda);
            }
        } else if (lambda < 5000) {
            const initial_pois = @ceil(lambda);
            const initial_mass = density(initial_pois, lambda);
            const initial_cumu = probability(initial_pois, lambda);
            for (buffer) |*x| {
                const uni = generator.float(f64);
                x.* = guidedSearch(uni, lambda, initial_pois, initial_mass, initial_cumu);
            }
        } else {
            const beta = std.math.pi / @sqrt(3 * lambda);
            const c = 0.801 - 0.687 / @sqrt(lambda);
            const k = @log(c) - lambda - @log(beta);
            for (buffer) |*x| {
                x.* = rejection(generator, lambda, beta, k);
            }
        }
        return buffer;
    }
};

fn linearSearch(p: f64, lambda: f64) f64 {
    var pois: f64 = 0;
    var mass = @exp(-lambda);
    var cumu = mass;
    while (cumu <= p) : (pois += 1) {
        mass *= lambda / (pois + 1);
        cumu += mass;
    }
    return pois;
}

fn guidedSearch(p: f64, lambda: f64, initial_pois: f64, initial_mass: f64, initial_cumu: f64) f64 {
    var pois = initial_pois;
    var mass = initial_mass;
    var cumu = initial_cumu;
    if (initial_cumu <= p) {
        while (cumu <= p) {
            pois += 1;
            mass *= lambda / pois;
            cumu += mass;
        }
    } else {
        while (true) {
            cumu -= mass;
            mass *= pois / lambda;
            if (cumu <= p) {
                break;
            }
            pois -= 1;
        }
    }
    return pois;
}

fn rejection(generator: std.rand.Random, lambda: f64, beta: f64, k: f64) f64 {
    while (true) {
        const uni1 = generator.float(f64);
        const z = @log((1 - uni1) / uni1);
        const x = lambda - z / beta;
        if (x < -0.5) {
            continue;
        }
        const n = @round(x + 0.5);
        const uni2 = generator.float(f64);
        const ezp1 = @exp(z) + 1;
        const left = z + @log(uni2 / (ezp1 * ezp1));
        const right = k + n * @log(lambda) - std.math.lgamma(f64, n + 1);
        if (left <= right) {
            return n;
        }
    }
}

const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

// zig fmt: off
test "poisson.density" {
    try expectEqual(0, density(-inf, 3));
    try expectEqual(0, density( inf, 3));

    try expectApproxEqRel(0                 , density(-0.1, 3), eps);
    try expectApproxEqRel(0.0497870683678639, density( 0  , 3), eps);
    try expectApproxEqRel(0                 , density( 0.1, 3), eps);
    try expectApproxEqRel(0                 , density( 0.9, 3), eps);
    try expectApproxEqRel(0.1493612051035919, density( 1  , 3), eps);
    try expectApproxEqRel(0                 , density( 1.1, 3), eps);
}

test "poisson.probability" {
    try expectEqual(0, probability(-inf, 3));
    try expectEqual(1, probability( inf, 3));

    try expectApproxEqRel(0                 , probability(-0.1, 3), eps);
    try expectApproxEqRel(0.0497870683678639, probability( 0  , 3), eps);
    try expectApproxEqRel(0.0497870683678639, probability( 0.1, 3), eps);
    try expectApproxEqRel(0.0497870683678639, probability( 0.9, 3), eps);
    try expectApproxEqRel(0.1991482734714558, probability( 1  , 3), eps);
    try expectApproxEqRel(0.1991482734714558, probability( 1.1, 3), eps);
}

test "poisson.quantile" {
    try expectEqual(  0, quantile(0                 , 3));
    try expectEqual(  0, quantile(0.0497870683678638, 3));
    try expectEqual(  0, quantile(0.0497870683678639, 3));
    try expectEqual(  1, quantile(0.0497870683678640, 3));
    try expectEqual(  1, quantile(0.1991482734714556, 3));
    try expectEqual(  1, quantile(0.1991482734714557, 3));
    try expectEqual(  2, quantile(0.1991482734714558, 3));
    try expectEqual(inf, quantile(1                 , 3));
}

test "poisson.random.single" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(2, random.single(gen, 3));
    try expectEqual(2, random.single(gen, 3));
    try expectEqual(3, random.single(gen, 3));
}
