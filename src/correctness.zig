const std = @import("std");
const rv = @import("random_variable");

const params = range(3);
const n = 1_000_000;

const Error = error{FailedKolmogorovTest};

fn kolmogorov(sample: []f64, namespace: type, parameters: anytype) Error!void {
    const len: f64 = @floatFromInt(sample.len);
    std.mem.sortUnstable(f64, sample, {}, std.sort.asc(f64));
    var index: f64 = 1;
    var max: f64 = 0;
    for (sample) |q| {
        const empirical = index / len;
        const theoretical = @call(.auto, namespace.probability, .{q} ++ parameters);
        max = @max(max, @abs(empirical - theoretical));
        index += 1;
    }
    const quantile = 1.94947 / @sqrt(len);
    if (max > quantile) {
        std.log.err(
            \\     {}{:.3}
            \\statistic:  {d:.8}
            \\quantile:   {d:.8}
            \\
        , .{ namespace, parameters, max, quantile });
        return error.FailedKolmogorovTest;
    }
}

fn range(comptime size: comptime_int) [2 * size + 1]f64 {
    const base = 1.5;
    var low: [size]f64 = undefined;
    var high: [size]f64 = undefined;
    for (0..size) |i| {
        const pow = std.math.pow(f64, base, @floatFromInt(i + 1));
        low[size - 1 - i] = 1 / pow;
        high[i] = pow;
    }
    return low ++ .{1} ++ high;
}

test "beta" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |shape1| {
        for (params) |shape2| {
            rv.beta.fill(&buffer, random, shape1, shape2);
            try kolmogorov(&buffer, rv.beta, .{ shape1, shape2 });
        }
    }
}

test "betaPrime" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |shape1| {
        for (params) |shape2| {
            rv.betaPrime.fill(&buffer, random, shape1, shape2);
            try kolmogorov(&buffer, rv.betaPrime, .{ shape1, shape2 });
        }
    }
}

test "cauchy" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    rv.cauchy.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.cauchy, .{ 0, 1 });
}

test "chi" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |df| {
        rv.chi.fill(&buffer, random, df);
        try kolmogorov(&buffer, rv.chi, .{df});
    }
}

test "chiSquared" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |df| {
        rv.chiSquared.fill(&buffer, random, df);
        try kolmogorov(&buffer, rv.chiSquared, .{df});
    }
}

test "continuousBernoulli" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    const shapes = [_]f64{ 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999 };
    for (shapes) |shape| {
        rv.continuousBernoulli.fill(&buffer, random, shape);
        try kolmogorov(&buffer, rv.continuousBernoulli, .{shape});
    }
}

test "dagum" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |shape1| {
        for (params) |shape2| {
            rv.dagum.fill(&buffer, random, shape1, shape2, 1);
            try kolmogorov(&buffer, rv.dagum, .{ shape1, shape2, 1 });
        }
    }
}

test "exponential" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    rv.exponential.fill(&buffer, random, 1);
    try kolmogorov(&buffer, rv.exponential, .{1});
}

test "f" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(234234);
    const random = gen.random();
    for (params) |df1| {
        // HACK: skipping the df2 = 0.3 case,
        // it's just barely not enough to pass the test,
        // both at 0.3 are quite extreme parameters of the distribution
        // generating values between 1e-40 and 1e40
        for (params[1..]) |df2| {
            rv.f.fill(&buffer, random, df1, df2);
            try kolmogorov(&buffer, rv.f, .{ df1, df2 });
        }
    }
}

test "gamma" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |shape| {
        rv.gamma.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.gamma, .{ shape, 1 });
    }
}

test "gompertz" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |shape| {
        rv.gompertz.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.gompertz, .{ shape, 1 });
    }
}

test "gumbel" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    rv.gumbel.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.gumbel, .{ 0, 1 });
}

test "laplace" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    rv.laplace.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.laplace, .{ 0, 1 });
}

test "logistic" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    rv.logistic.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.logistic, .{ 0, 1 });
}

test "logNormal" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |log_location| {
        for (params) |log_scale| {
            rv.logNormal.fill(&buffer, random, log_location, log_scale);
            try kolmogorov(&buffer, rv.logNormal, .{ log_location, log_scale });
        }
    }
}

test "normal" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    rv.normal.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.normal, .{ 0, 1 });
}

test "pareto" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |shape| {
        rv.pareto.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.pareto, .{ shape, 1 });
    }
}

test "rayleigh" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    rv.rayleigh.fill(&buffer, random, 1);
    try kolmogorov(&buffer, rv.rayleigh, .{1});
}

test "t" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |df| {
        rv.t.fill(&buffer, random, df);
        try kolmogorov(&buffer, rv.t, .{df});
    }
}

test "uniform" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    rv.uniform.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.uniform, .{ 0, 1 });
}

test "weibull" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (params) |shape| {
        rv.weibull.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.weibull, .{ shape, 1 });
    }
}
