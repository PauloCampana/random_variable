//! Generators for Random Variables with different distributions.

const std = @import("std");
const math = std.math;
const RandomVariable = @This();

// fields in struct

generator: std.rand.Random,
comptime T: type = f64,

// functions you can call with rv.fun()

pub fn uniform(self: *RandomVariable, min: f64, max: f64) f64 {
    const uni = self.generator.float(self.T);
    return min + (max - min) * uni;
}

pub fn bernoulli(self: *RandomVariable, prob: f64) bool {
    const uni = self.generator.float(self.T);
    const ber = uni < prob;
    return ber;
}

pub fn geometric(self: *RandomVariable, prob: f64) u64 {
    const uni = self.generator.float(self.T);
    const geo = @floor(@log(uni) / math.log1p(-prob));
    return @intFromFloat(geo);
}

pub fn poisson(self: *RandomVariable, lambda: f64) u64 {
    const uni = self.generator.float(self.T);
    var p = @exp(-lambda);
    var f = p;
    var poi: u64 = 0;
    while (uni >= f) : (poi += 1) {
        p *= lambda / @as(f64, poi + 1);
        f += p;
    }
    return poi;
}

pub fn binomial(self: *RandomVariable, size: u64, prob: f64) u64 {
    var bin: u64 = 0;
    for (0..size) |_| {
        bin += @as(u64, bernoulli(self, prob));
    }
    return bin;
}

pub fn negative_binomial(self: *RandomVariable, size: u64, prob: f64) u64 {
    var nbin: u64 = 0;
    for (0..size) |_| {
        nbin += geometric(self, prob);
    }
    return nbin;
}

pub fn exponential(self: *RandomVariable, rate: f64) f64 {
    const exp = self.generator.floatExp(f64) / rate;
    return exp;
}

pub fn weibull(self: *RandomVariable, shape: f64, rate: f64) f64 {
    const uni = self.generator.float(self.T);
    const wei = math.pow(f64, -@log(uni), 1 / shape) / rate;
    return wei;
}

pub fn cauchy(self: *RandomVariable, location: f64, scale: f64) f64 {
    const uni = self.generator.float(self.T);
    const cau = location + scale * @tan(math.pi * uni);
    return cau;
}

pub fn logistic(self: *RandomVariable, location: f64, scale: f64) f64 {
    const uni = self.generator.float(self.T);
    const log = location + scale * @log(uni / (1 - uni));
    return log;
}

pub fn gamma(self: *RandomVariable, shape: f64, rate: f64) f64 {
    const correct = shape >= 1;
    const d1 = shape - 1.0 / 3.0;
    const d = if (correct) d1 else d1 + 1;
    const c = 1 / @sqrt(9 * d);

    var v: f64 = undefined;
    while (true) {
        var z: f64 = undefined;
        while (true) {
            z = normal(self, 0, 1);
            v = 1 + c * z;
            if (v > 0) {
                break;
            }
        }
        v *= v * v;
        z *= z;
        const uni = self.generator.float(self.T);
        if (
            uni < 1 - 0.0331 * z * z or
            @log(uni) < 0.5 * z + d * (1 - v + @log(v))
        ) {
            break;
        }
    }
    const gam = d * v;

    if (correct) {
        return gam / rate;
    } else {
        const uni = self.generator.float(self.T);
        const correction = math.pow(f64, uni, 1 / shape);
        return gam * correction / rate;
    }
}

pub fn chi_squared(self: *RandomVariable, df: f64) f64 {
    return gamma(self, 0.5 * df, 0.5);
}

pub fn F(self: *RandomVariable, df1: f64, df2: f64) f64 {
    const chi1 = chi_squared(self, df1);
    const chi2 = chi_squared(self, df2);
    const f = chi1 / chi2 / df1 * df2;
    return f;
}

pub fn beta(self: *RandomVariable, shape1: f64, shape2: f64) f64 {
    const gam1 = gamma(self, shape1, 1);
    const gam2 = gamma(self, shape2, 1);
    const bet = gam1 / (gam1 + gam2);
    return bet;
}

pub fn normal(self: *RandomVariable, mean: f64, sd: f64) f64 {
    const nor = self.generator.floatNorm(f64);
    return mean + sd * nor;
}

pub fn lognormal(self: *RandomVariable, meanlog: f64, sdlog: f64) f64 {
    return @exp(normal(self, meanlog, sdlog));
}

pub fn t(self: *RandomVariable, df: f64) f64 {
    const nor = normal(self, 0, 1);
    const chi = chi_squared(self, df);
    const T = nor * @sqrt(df / chi);
    return T;
}

// TODO: remove f64s, use T
// TODO: bounds check for parameters
// TODO: special cases of the distros
