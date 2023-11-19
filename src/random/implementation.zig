const std = @import("std");
const expectEqual = @import("../thirdyparty/testing.zig").expectEqual;
const expectApproxEqRel = @import("../thirdyparty/testing.zig").expectApproxEqRel;

pub fn uniform(comptime C: type, generator: std.rand.Random, min: C, max: C) C {
    const uni = generator.float(C);
    return min + (max - min) * uni;
}

test "random.uniform" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(uniform(f64, gen, 0, 1), 0x1.75d61490b23dfp-2);
    try expectApproxEqRel(uniform(f64, gen, 0, 1), 0x1.a6f3dc380d507p-2);
    try expectApproxEqRel(uniform(f64, gen, 0, 1), 0x1.fdf91ec9a7bfcp-2);
    try expectApproxEqRel(uniform(f64, gen, 1, 1), 0x1.0000000000000p+0);
}

pub fn bernoulli(comptime D: type, comptime C: type, generator: std.rand.Random, prob: C) D {
    const uni = generator.float(C);
    const ber = @intFromBool(uni < prob);
    return std.math.lossyCast(D, ber);
}

test "random.bernoulli" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(bernoulli(u64, f64, gen, 0.2), 0);
    try expectEqual(bernoulli(u64, f64, gen, 0.2), 0);
    try expectEqual(bernoulli(u64, f64, gen, 0.2), 0);
    try expectEqual(bernoulli(u64, f64, gen, 0  ), 0);
    try expectEqual(bernoulli(u64, f64, gen, 1  ), 1);
}

pub fn geometric(comptime D: type, comptime C: type, generator: std.rand.Random, prob: C) D {
    const rate = -std.math.log1p(-prob);
    const exp = generator.floatExp(C);
    const geo = @trunc(exp / rate);
    return std.math.lossyCast(D, geo);
}

test "random.geometric" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(geometric(u64, f64, gen, 0.2), 0);
    try expectEqual(geometric(u64, f64, gen, 0.2), 9);
    try expectEqual(geometric(u64, f64, gen, 0.2), 0);
    try expectEqual(geometric(u64, f64, gen, 1  ), 0);
}

pub fn poisson(comptime D: type, comptime C: type, generator: std.rand.Random, lambda: C) D {
    const uni = generator.float(C);
    var p = @exp(-lambda);
    var f = p;
    var poi: C = 1;
    while (uni >= f) : (poi += 1) {
        p *= lambda / poi;
        f += p;
    }
    return std.math.lossyCast(D, poi - 1);
}

test "random.poisson" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(poisson(u64, f64, gen, 3), 2);
    try expectEqual(poisson(u64, f64, gen, 3), 2);
    try expectEqual(poisson(u64, f64, gen, 3), 3);
}

pub fn binomial(comptime D: type, comptime C: type, generator: std.rand.Random, size: usize, prob: C) D {
    if (prob == 1) {
        return std.math.lossyCast(D, size);
    }
    const uni = generator.float(C);
    const n = @as(C, @floatFromInt(size));
    const np1 = n + 1;
    const q = 1 - prob;
    const pq = prob / q;
    var p = std.math.pow(C, q, n);
    var f = p;
    var bin: C = 1;
    while (uni >= f) : (bin += 1) {
        p *= pq * (np1 - bin) / bin;
        f += p;
    }
    return std.math.lossyCast(D, bin - 1);
}

test "random.binomial" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(binomial(u64, f64, gen, 10, 0.2), 1 );
    try expectEqual(binomial(u64, f64, gen, 10, 0.2), 2 );
    try expectEqual(binomial(u64, f64, gen, 10, 0.2), 2 );
    try expectEqual(binomial(u64, f64, gen, 0 , 0.2), 0 );
    try expectEqual(binomial(u64, f64, gen, 10, 0  ), 0 );
    try expectEqual(binomial(u64, f64, gen, 10, 1  ), 10);
}

pub fn negativeBinomial(comptime D: type, comptime C: type, generator: std.rand.Random, size: usize, prob: C) D {
    const uni = generator.float(C);
    const n = @as(C, @floatFromInt(size));
    const nm1 = n - 1;
    const q = 1 - prob;
    var p = std.math.pow(C, prob, n);
    var f = p;
    var nbi: C = 1;
    while (uni >= f) : (nbi += 1) {
        p *= q * (nm1 + nbi) / nbi;
        f += p;
    }
    return std.math.lossyCast(D, nbi - 1);
}

test "random.negativeBinomial" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(negativeBinomial(u64, f64, gen, 10, 0.2), 34);
    try expectEqual(negativeBinomial(u64, f64, gen, 10, 0.2), 36);
    try expectEqual(negativeBinomial(u64, f64, gen, 10, 0.2), 38);
    try expectEqual(negativeBinomial(u64, f64, gen, 10, 1  ), 0 );
}

pub fn exponential(comptime C: type, generator: std.rand.Random, rate: C) C {
    const exp = generator.floatExp(C);
    return exp / rate;
}

test "random.exponential" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(exponential(f64, gen, 3), 0x1.0d10389b44e27p-4);
    try expectApproxEqRel(exponential(f64, gen, 3), 0x1.65addca068349p-1);
    try expectApproxEqRel(exponential(f64, gen, 3), 0x1.444f149040ffap-6);
}

pub fn weibull(comptime C: type, generator: std.rand.Random, shape: C, rate: C) C {
    const exp = generator.floatExp(C);
    const wei = std.math.pow(C, exp, 1 / shape);
    return wei / rate;
}

test "random.weibull" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(weibull(f64, gen, 3, 0.5), 0x1.29f2f11294770p+0);
    try expectApproxEqRel(weibull(f64, gen, 3, 0.5), 0x1.479bbb94bd291p+1);
    try expectApproxEqRel(weibull(f64, gen, 3, 0.5), 0x1.8f80c328506e1p-1);
}

pub fn cauchy(comptime C: type, generator: std.rand.Random, location: C, scale: C) C {
    const uni = generator.float(C);
    return location + scale * @tan(std.math.pi * uni);
}

test "random.cauchy" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(cauchy(f64, gen, 0, 1), 0x1.1baa5d88fd11ap+1);
    try expectApproxEqRel(cauchy(f64, gen, 0, 1), 0x1.c8d1141faf950p+1);
    try expectApproxEqRel(cauchy(f64, gen, 0, 1), 0x1.419f9beb83432p+7);
}

pub fn logistic(comptime C: type, generator: std.rand.Random, location: C, scale: C) C {
    const uni = generator.float(C);
    return location + scale * @log(uni / (1 - uni));
}

test "random.logistic" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(logistic(f64, gen, 0, 1), -0x1.1b5814cb6fc9ap-1);
    try expectApproxEqRel(logistic(f64, gen, 0, 1), -0x1.67d902cb3c67ep-2);
    try expectApproxEqRel(logistic(f64, gen, 0, 1), -0x1.0370f3fe2a1a1p-7);
}

pub fn gamma(comptime C: type, generator: std.rand.Random, shape: C, rate: C) C {
    const correct = shape >= 1;
    const d = blk: {
        const d0 = shape - 1.0 / 3.0;
        break :blk if (shape >= 1) d0 else d0 + 1;
    };
    const c = 1 / (3 * @sqrt(d));
    const gam = whl: while (true) {
        var v: C = undefined;
        var z: C = undefined;
        while (true) {
            z = generator.floatNorm(C);
            v = 1 + c * z;
            if (v > 0) {
                break;
            }
        }
        v *= v * v;
        z *= z;
        const uni = generator.float(C);
        const cond0 = uni < 1 - 0.0331 * z * z;
        if (cond0 or @log(uni) < 0.5 * z + d * (1 - v + @log(v))) {
            break :whl d * v;
        }
    };
    if (correct) {
        return gam / rate;
    } else {
        const uni = generator.float(C);
        const correction = std.math.pow(C, uni, 1 / shape);
        return gam / rate * correction;
    }
}

test "random.gamma" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(gamma(f64, gen, 3, 5), 0x1.c5f1fac1e8796p-2);
    try expectApproxEqRel(gamma(f64, gen, 3, 5), 0x1.ffa96ffd15766p-2);
    try expectApproxEqRel(gamma(f64, gen, 3, 5), 0x1.0ff0a4d0472aap-1);
}

pub fn chiSquared(comptime C: type, generator: std.rand.Random, df: C) C {
    return gamma(C, generator, 0.5 * df, 0.5);
}

test "random.chiSquared" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(chiSquared(f64, gen, 3), 0x1.c198f554d3db5p+0);
    try expectApproxEqRel(chiSquared(f64, gen, 3), 0x1.0e7afeee50b89p+1);
    try expectApproxEqRel(chiSquared(f64, gen, 3), 0x1.28ce118715efcp+1);
}

pub fn F(comptime C: type, generator: std.rand.Random, df1: C, df2: C) C {
    const chinum = chiSquared(C, generator, df1);
    const chiden = chiSquared(C, generator, df2);
    return chinum / chiden * df2 / df1;
}

test "random.F" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(F(f64, gen, 3, 5), 0x1.73d1aa315be37p-1);
    try expectApproxEqRel(F(f64, gen, 3, 5), 0x1.bf5ec1a08f87bp-2);
    try expectApproxEqRel(F(f64, gen, 3, 5), 0x1.cbddabd676b5fp-1);
}

pub fn beta(comptime C: type, generator: std.rand.Random, shape1: C, shape2: C) C {
    const gam1 = gamma(C, generator, shape1, 1);
    const gam2 = gamma(C, generator, shape2, 1);
    return gam1 / (gam1 + gam2);
}

test "random.beta" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(beta(f64, gen, 3, 5), 0x1.54d531aa6eb30p-2);
    try expectApproxEqRel(beta(f64, gen, 3, 5), 0x1.05f28586a9fadp-2);
    try expectApproxEqRel(beta(f64, gen, 3, 5), 0x1.77ac6b3ffb648p-2);
}

pub fn normal(comptime C: type, generator: std.rand.Random, mean: C, sd: C) C {
    const nor = generator.floatNorm(C);
    return mean + sd * nor;
}

test "random.normal" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(normal(f64, gen, 0, 1), -0x1.2bd4a0beac2dfp-2);
    try expectApproxEqRel(normal(f64, gen, 0, 1), -0x1.6d1e253ea4858p-1);
    try expectApproxEqRel(normal(f64, gen, 0, 1), -0x1.af653db4b3107p-4);
}

pub fn logNormal(comptime C: type, generator: std.rand.Random, meanlog: C, sdlog: C) C {
    const nor = generator.floatNorm(C);
    const log = meanlog + sdlog * nor;
    return @exp(log);
}

test "random.logNormal" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(logNormal(f64, gen, 0, 1), 0x1.7e09d992a530ep-1);
    try expectApproxEqRel(logNormal(f64, gen, 0, 1), 0x1.f5e0036c64e29p-2);
    try expectApproxEqRel(logNormal(f64, gen, 0, 1), 0x1.ccd17150549b1p-1);
}

pub fn t(comptime C: type, generator: std.rand.Random, df: C) C {
    if (df == 1) {
        return cauchy(C, generator, 0, 1);
    }
    const nor = generator.floatNorm(C);
    const chi = chiSquared(C, generator, df);
    return nor * @sqrt(df / chi);
}

test "random.t" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(t(f64, gen, 3), -0x1.ed977ce651337p-2);
    try expectApproxEqRel(t(f64, gen, 3), -0x1.62bae37cf8d83p+1);
    try expectApproxEqRel(t(f64, gen, 3),  0x1.a5797fad46fcap-1);
}
