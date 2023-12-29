const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const eps = 10 * std.math.floatEps(f64); // 2.22 × 10^-15

pub fn uniform(generator: std.rand.Random, min: f64, max: f64) f64 {
    const uni = generator.float(f64);
    return min + (max - min) * uni;
}

test "random.uniform" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, 0x1.75d61490b23dfp-2), uniform(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0x1.a6f3dc380d507p-2), uniform(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0x1.fdf91ec9a7bfcp-2), uniform(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0x1.0000000000000p+0), uniform(gen, 1, 1), eps);
}

pub fn bernoulli(generator: std.rand.Random, prob: f64) f64 {
    const uni = generator.float(f64);
    const ber = @intFromBool(uni < prob);
    return @floatFromInt(ber);
}

test "random.bernoulli" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(@as(f64, 0), bernoulli(gen, 0.2));
    try expectEqual(@as(f64, 0), bernoulli(gen, 0.2));
    try expectEqual(@as(f64, 0), bernoulli(gen, 0.2));
    try expectEqual(@as(f64, 0), bernoulli(gen, 0  ));
    try expectEqual(@as(f64, 1), bernoulli(gen, 1  ));
}

pub fn geometric(generator: std.rand.Random, prob: f64) f64 {
    const rate = -std.math.log1p(-prob);
    const exp = generator.floatExp(f64);
    return @trunc(exp / rate);
}

test "random.geometric" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(@as(f64, 0), geometric(gen, 0.2));
    try expectEqual(@as(f64, 9), geometric(gen, 0.2));
    try expectEqual(@as(f64, 0), geometric(gen, 0.2));
    try expectEqual(@as(f64, 0), geometric(gen, 1  ));
}

pub fn poisson(generator: std.rand.Random, lambda: f64) f64 {
    const uni = generator.float(f64);
    var mass = @exp(-lambda);
    var cumu = mass;
    var poi: f64 = 1;
    while (uni >= cumu) : (poi += 1) {
        mass *= lambda / poi;
        cumu += mass;
    }
    return poi - 1;
}

test "random.poisson" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(@as(f64, 2), poisson(gen, 3));
    try expectEqual(@as(f64, 2), poisson(gen, 3));
    try expectEqual(@as(f64, 3), poisson(gen, 3));
}

pub fn binomial(generator: std.rand.Random, size: u64, prob: f64) f64 {
    if (prob == 1) {
        return @floatFromInt(size);
    }
    const uni = generator.float(f64);
    const n = @as(f64, @floatFromInt(size));
    const np1 = n + 1;
    const qrob = 1 - prob;
    const pq = prob / qrob;
    var mass = std.math.pow(f64, qrob, n);
    var cumu = mass;
    var bin: f64 = 1;
    while (uni >= cumu) : (bin += 1) {
        mass *= pq * (np1 - bin) / bin;
        cumu += mass;
    }
    return bin - 1;
}

test "random.binomial" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(@as(f64, 1 ), binomial(gen, 10, 0.2));
    try expectEqual(@as(f64, 2 ), binomial(gen, 10, 0.2));
    try expectEqual(@as(f64, 2 ), binomial(gen, 10, 0.2));
    try expectEqual(@as(f64, 0 ), binomial(gen, 0 , 0.2));
    try expectEqual(@as(f64, 0 ), binomial(gen, 10, 0  ));
    try expectEqual(@as(f64, 10), binomial(gen, 10, 1  ));
}

pub fn negativeBinomial(generator: std.rand.Random, size: u64, prob: f64) f64 {
    const uni = generator.float(f64);
    const n = @as(f64, @floatFromInt(size));
    const nm1 = n - 1;
    const qrob = 1 - prob;
    var mass = std.math.pow(f64, prob, n);
    var cumu = mass;
    var nbi: f64 = 1;
    while (uni >= cumu) : (nbi += 1) {
        mass *= qrob * (nm1 + nbi) / nbi;
        cumu += mass;
    }
    return nbi - 1;
}

test "random.negativeBinomial" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectEqual(@as(f64, 34), negativeBinomial(gen, 10, 0.2));
    try expectEqual(@as(f64, 36), negativeBinomial(gen, 10, 0.2));
    try expectEqual(@as(f64, 38), negativeBinomial(gen, 10, 0.2));
    try expectEqual(@as(f64, 0 ), negativeBinomial(gen, 10, 1  ));
}

pub fn exponential(generator: std.rand.Random, rate: f64) f64 {
    const exp = generator.floatExp(f64);
    return exp / rate;
}

test "random.exponential" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, 0x1.0d10389b44e27p-4), exponential(gen, 3), eps);
    try expectApproxEqRel(@as(f64, 0x1.65addca068349p-1), exponential(gen, 3), eps);
    try expectApproxEqRel(@as(f64, 0x1.444f149040ffap-6), exponential(gen, 3), eps);
}

pub fn weibull(generator: std.rand.Random, shape: f64, rate: f64) f64 {
    const exp = generator.floatExp(f64);
    const wei = std.math.pow(f64, exp, 1 / shape);
    return wei / rate;
}

test "random.weibull" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, 0x1.29f2f11294770p+0), weibull(gen, 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 0x1.479bbb94bd291p+1), weibull(gen, 3, 0.5), eps);
    try expectApproxEqRel(@as(f64, 0x1.8f80c328506e1p-1), weibull(gen, 3, 0.5), eps);
}

pub fn cauchy(generator: std.rand.Random, location: f64, scale: f64) f64 {
    const uni = generator.float(f64);
    return location + scale * @tan(std.math.pi * uni);
}

test "random.cauchy" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, 0x1.1baa5d88fd11ap+1), cauchy(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0x1.c8d1141faf950p+1), cauchy(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0x1.419f9beb83432p+7), cauchy(gen, 0, 1), eps);
}

pub fn logistic(generator: std.rand.Random, location: f64, scale: f64) f64 {
    const uni = generator.float(f64);
    return location + scale * @log(uni / (1 - uni));
}

test "random.logistic" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, -0x1.1b5814cb6fc9ap-1), logistic(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, -0x1.67d902cb3c67ep-2), logistic(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, -0x1.0370f3fe2a1a1p-7), logistic(gen, 0, 1), eps);
}

pub fn gamma(generator: std.rand.Random, shape: f64, rate: f64) f64 {
    const correct = shape >= 1;
    const increment: f64 = if (correct) 0 else 1;
    const d = shape - 1.0 / 3.0 + increment;
    const c = 1 / (3 * @sqrt(d));
    const gam = blk: while (true) {
        var v: f64 = undefined;
        var z: f64 = undefined;
        while (true) {
            z = generator.floatNorm(f64);
            v = 1 + c * z;
            if (v > 0) {
                break;
            }
        }
        v *= v * v;
        z *= z;
        const uni = generator.float(f64);
        const cond0 = uni < 1 - 0.0331 * z * z;
        if (cond0 or @log(uni) < 0.5 * z + d * (1 - v + @log(v))) {
            break :blk d * v;
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

test "random.gamma" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, 0x1.c5f1fac1e8796p-2), gamma(gen, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0x1.ffa96ffd15766p-2), gamma(gen, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0x1.0ff0a4d0472aap-1), gamma(gen, 3, 5), eps);
}

pub fn chiSquared(generator: std.rand.Random, df: f64) f64 {
    return gamma(generator, 0.5 * df, 0.5);
}

test "random.chiSquared" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, 0x1.c198f554d3db5p+0), chiSquared(gen, 3), eps);
    try expectApproxEqRel(@as(f64, 0x1.0e7afeee50b89p+1), chiSquared(gen, 3), eps);
    try expectApproxEqRel(@as(f64, 0x1.28ce118715efcp+1), chiSquared(gen, 3), eps);
}

pub fn F(generator: std.rand.Random, df1: f64, df2: f64) f64 {
    const chinum = gamma(generator, 0.5 * df1, 1);
    const chiden = gamma(generator, 0.5 * df2, 1);
    // const chinum = chiSquared(generator, df1);
    // const chiden = chiSquared(generator, df2);
    return chinum / chiden * df2 / df1;
}

test "random.F" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, 0x1.73d1aa315be37p-1), F(gen, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0x1.bf5ec1a08f87bp-2), F(gen, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0x1.cbddabd676b5fp-1), F(gen, 3, 5), eps);
}

pub fn beta(generator: std.rand.Random, shape1: f64, shape2: f64) f64 {
    const gam1 = gamma(generator, shape1, 1);
    const gam2 = gamma(generator, shape2, 1);
    return gam1 / (gam1 + gam2);
}

test "random.beta" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, 0x1.54d531aa6eb30p-2), beta(gen, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0x1.05f28586a9fadp-2), beta(gen, 3, 5), eps);
    try expectApproxEqRel(@as(f64, 0x1.77ac6b3ffb648p-2), beta(gen, 3, 5), eps);
}

pub fn normal(generator: std.rand.Random, mean: f64, sd: f64) f64 {
    const nor = generator.floatNorm(f64);
    return mean + sd * nor;
}

test "random.normal" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, -0x1.2bd4a0beac2dfp-2), normal(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, -0x1.6d1e253ea4858p-1), normal(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, -0x1.af653db4b3107p-4), normal(gen, 0, 1), eps);
}

pub fn logNormal(generator: std.rand.Random, meanlog: f64, sdlog: f64) f64 {
    const nor = generator.floatNorm(f64);
    const log = meanlog + sdlog * nor;
    return @exp(log);
}

test "random.logNormal" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, 0x1.7e09d992a530ep-1), logNormal(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0x1.f5e0036c64e29p-2), logNormal(gen, 0, 1), eps);
    try expectApproxEqRel(@as(f64, 0x1.ccd17150549b1p-1), logNormal(gen, 0, 1), eps);
}

pub fn t(generator: std.rand.Random, df: f64) f64 {
    if (df == 1) {
        const uni = generator.float(f64);
        return @tan(std.math.pi * uni);
    }
    const nor = generator.floatNorm(f64);
    const chi = chiSquared(generator, df);
    return nor * @sqrt(df / chi);
}

test "random.t" {
    var prng = std.rand.DefaultPrng.init(0);
    const gen = prng.random();
    try expectApproxEqRel(@as(f64, -0x1.ed977ce651337p-2), t(gen, 3), eps);
    try expectApproxEqRel(@as(f64, -0x1.62bae37cf8d83p+1), t(gen, 3), eps);
    try expectApproxEqRel(@as(f64,  0x1.a5797fad46fcap-1), t(gen, 3), eps);
}
