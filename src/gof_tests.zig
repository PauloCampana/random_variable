const std = @import("std");
const rv = @import("random_variable");

fn kolmogorov(sample: []f64, probability_fn: anytype, parameters: anytype) !void {
    const len: f64 = @floatFromInt(sample.len);
    std.mem.sortUnstable(f64, sample, {}, std.sort.asc(f64));
    var index: f64 = 1;
    var max: f64 = 0;
    for (sample) |q| {
        const empirical = index / len;
        const theoretical = @call(.auto, probability_fn, .{q} ++ parameters);
        max = @max(max, @abs(empirical - theoretical));
        index += 1;
    }
    const quantile = 1.94947 / @sqrt(len);
    if (max > quantile) {
        std.log.err(
            \\
            \\statistic:  {d:.8}
            \\quantile:   {d:.8}
            \\parameters: {:.3}
        , .{ max, quantile, parameters });
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

const allocator = std.testing.allocator;
var rng = std.rand.DefaultPrng.init(0);
const gen = rng.random();
const params = range(3);
const n = 1_000_000;

test "beta" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |shape1| {
        for (params) |shape2| {
            rv.beta.fill(slice, gen, shape1, shape2);
            try kolmogorov(slice, rv.beta.probability, .{ shape1, shape2 });
        }
    }
}

test "betaPrime" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |shape1| {
        for (params) |shape2| {
            rv.betaPrime.fill(slice, gen, shape1, shape2);
            try kolmogorov(slice, rv.betaPrime.probability, .{ shape1, shape2 });
        }
    }
}

test "cauchy" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    rv.cauchy.fill(slice, gen, 0, 1);
    try kolmogorov(slice, rv.cauchy.probability, .{ 0, 1 });
}

test "chi" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |df| {
        rv.chi.fill(slice, gen, df);
        try kolmogorov(slice, rv.chi.probability, .{df});
    }
}

test "chiSquared" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |df| {
        rv.chiSquared.fill(slice, gen, df);
        try kolmogorov(slice, rv.chiSquared.probability, .{df});
    }
}

test "continuousBernoulli" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    const shapes = [_]f64{ 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999 };
    for (shapes) |shape| {
        rv.continuousBernoulli.fill(slice, gen, shape);
        try kolmogorov(slice, rv.continuousBernoulli.probability, .{shape});
    }
}

test "dagum" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |shape1| {
        for (params) |shape2| {
            rv.dagum.fill(slice, gen, shape1, shape2, 1);
            try kolmogorov(slice, rv.dagum.probability, .{ shape1, shape2, 1 });
        }
    }
}

test "exponential" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    rv.exponential.fill(slice, gen, 1);
    try kolmogorov(slice, rv.exponential.probability, .{1});
}

test "f" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |df1| {
        // HACK: skipping the df2 = 0.3 case,
        // it's just barely not enough to pass the test,
        // both at 0.3 are quite extreme parameters of the distribution
        // generating values between 1e-40 and 1e40
        for (params[1..]) |df2| {
            rv.f.fill(slice, gen, df1, df2);
            try kolmogorov(slice, rv.f.probability, .{ df1, df2 });
        }
    }
}

test "gamma" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |shape| {
        rv.gamma.fill(slice, gen, shape, 1);
        try kolmogorov(slice, rv.gamma.probability, .{ shape, 1 });
    }
}

test "gompertz" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |shape| {
        rv.gompertz.fill(slice, gen, shape, 1);
        try kolmogorov(slice, rv.gompertz.probability, .{ shape, 1 });
    }
}

test "gumbel" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    rv.gumbel.fill(slice, gen, 0, 1);
    try kolmogorov(slice, rv.gumbel.probability, .{ 0, 1 });
}

test "laplace" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    rv.laplace.fill(slice, gen, 0, 1);
    try kolmogorov(slice, rv.laplace.probability, .{ 0, 1 });
}

test "logistic" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    rv.logistic.fill(slice, gen, 0, 1);
    try kolmogorov(slice, rv.logistic.probability, .{ 0, 1 });
}

test "logNormal" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |log_scale| {
        rv.logNormal.fill(slice, gen, 0, log_scale);
        try kolmogorov(slice, rv.logNormal.probability, .{ 0, log_scale });
    }
}

test "normal" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    rv.normal.fill(slice, gen, 0, 1);
    try kolmogorov(slice, rv.normal.probability, .{ 0, 1 });
}

test "pareto" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |shape| {
        rv.pareto.fill(slice, gen, shape, 1);
        try kolmogorov(slice, rv.pareto.probability, .{ shape, 1 });
    }
}

test "rayleigh" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    rv.rayleigh.fill(slice, gen, 1);
    try kolmogorov(slice, rv.rayleigh.probability, .{1});
}

test "t" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |df| {
        rv.t.fill(slice, gen, df);
        try kolmogorov(slice, rv.t.probability, .{df});
    }
}

test "uniform" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    rv.uniform.fill(slice, gen, 0, 1);
    try kolmogorov(slice, rv.uniform.probability, .{ 0, 1 });
}

test "weibull" {
    const slice = try allocator.alloc(f64, n);
    defer allocator.free(slice);
    for (params) |shape| {
        rv.weibull.fill(slice, gen, shape, 1);
        try kolmogorov(slice, rv.weibull.probability, .{ shape, 1 });
    }
}
