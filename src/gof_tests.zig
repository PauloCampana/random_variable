const std = @import("std");
const rv = @import("random_variable");

fn kolmogorov(sample: []f64, probability_fn: anytype, parameters: anytype) !void {
    const len = @as(f64, @floatFromInt(sample.len));
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
        std.debug.print(" \n", .{});
        std.debug.print("statistic: {d:.5}\n", .{max});
        std.debug.print("quantile: {d:.5}\n", .{quantile});
        std.debug.print("parameters: {:.3}\n", .{parameters});
        std.debug.print("sample[0..10]: {e:.3}\n", .{sample[0..10]});
        return error.FailedKolmogorovTest;
    }
}

fn range(comptime size: comptime_int) [2 * size + 1]f64 {
    const base = 1.5;
    var low: [size]f64 = undefined;
    var high: [size]f64 = undefined;
    for (0..size) |i| {
        const f = @as(f64, @floatFromInt(i + 1));
        low[size - 1 - i] = std.math.pow(f64, base, -f);
        high[i] = std.math.pow(f64, base, f);
    }
    return low ++ .{1} ++ high;
}

var rng = std.rand.DefaultPrng.init(0);
const gen = rng.random();
const all = std.testing.allocator;
const n = 1_000_000;

test "beta" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const shapes = range(3);
    for (shapes) |shape1| {
        for (shapes) |shape2| {
            const sample = rv.beta.fill(slice, gen, shape1, shape2);
            try kolmogorov(sample, rv.beta.probability, .{ shape1, shape2 });
        }
    }
}

test "betaPrime" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const shapes = range(3);
    for (shapes) |shape1| {
        for (shapes) |shape2| {
            const sample = rv.betaPrime.fill(slice, gen, shape1, shape2);
            try kolmogorov(sample, rv.betaPrime.probability, .{ shape1, shape2 });
        }
    }
}

test "cauchy" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const sample = rv.cauchy.fill(slice, gen, 0, 1);
    try kolmogorov(sample, rv.cauchy.probability, .{ 0, 1 });
}

test "chi" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const dfs = range(3);
    for (dfs) |df| {
        const sample = rv.chi.fill(slice, gen, df);
        try kolmogorov(sample, rv.chi.probability, .{df});
    }
}

test "chiSquared" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const dfs = range(3);
    for (dfs) |df| {
        const sample = rv.chiSquared.fill(slice, gen, df);
        try kolmogorov(sample, rv.chiSquared.probability, .{df});
    }
}

test "continuousBernoulli" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const shapes = [_]f64{ 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999 };
    for (shapes) |shape| {
        const sample = rv.continuousBernoulli.fill(slice, gen, shape);
        try kolmogorov(sample, rv.continuousBernoulli.probability, .{shape});
    }
}

test "dagum" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const shapes = range(3);
    for (shapes) |shape1| {
        for (shapes) |shape2| {
            const sample = rv.dagum.fill(slice, gen, shape1, shape2, 1);
            try kolmogorov(sample, rv.dagum.probability, .{ shape1, shape2, 1 });
        }
    }
}

test "exponential" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const sample = rv.exponential.fill(slice, gen, 1);
    try kolmogorov(sample, rv.exponential.probability, .{1});
}

test "f" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const dfs = range(2);
    for (dfs) |df1| {
        for (dfs) |df2| {
            const sample = rv.f.fill(slice, gen, df1, df2);
            try kolmogorov(sample, rv.f.probability, .{ df1, df2 });
        }
    }
}

test "gamma" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const shapes = range(3);
    for (shapes) |shape| {
        const sample = rv.gamma.fill(slice, gen, shape, 1);
        try kolmogorov(sample, rv.gamma.probability, .{ shape, 1 });
    }
}

test "gompertz" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const shapes = range(3);
    for (shapes) |shape| {
        const sample = rv.gompertz.fill(slice, gen, shape, 1);
        try kolmogorov(sample, rv.gompertz.probability, .{ shape, 1 });
    }
}

test "gumbel" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const sample = rv.gumbel.fill(slice, gen, 0, 1);
    try kolmogorov(sample, rv.gumbel.probability, .{ 0, 1 });
}

test "laplace" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const sample = rv.laplace.fill(slice, gen, 0, 1);
    try kolmogorov(sample, rv.laplace.probability, .{ 0, 1 });
}

test "logistic" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const sample = rv.logistic.fill(slice, gen, 0, 1);
    try kolmogorov(sample, rv.logistic.probability, .{ 0, 1 });
}

test "logNormal" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const log_scales = range(3);
    for (log_scales) |log_scale| {
        const sample = rv.logNormal.fill(slice, gen, 0, log_scale);
        try kolmogorov(sample, rv.logNormal.probability, .{ 0, log_scale });
    }
}

test "normal" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const sample = rv.normal.fill(slice, gen, 0, 1);
    try kolmogorov(sample, rv.normal.probability, .{ 0, 1 });
}

test "pareto" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const shapes = range(3);
    for (shapes) |shape| {
        const sample = rv.pareto.fill(slice, gen, shape, 1);
        try kolmogorov(sample, rv.pareto.probability, .{ shape, 1 });
    }
}

test "rayleigh" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const sample = rv.rayleigh.fill(slice, gen, 1);
    try kolmogorov(sample, rv.rayleigh.probability, .{1});
}

test "t" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const dfs = range(3);
    for (dfs) |df| {
        const sample = rv.t.fill(slice, gen, df);
        try kolmogorov(sample, rv.t.probability, .{df});
    }
}

test "uniform" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const sample = rv.uniform.fill(slice, gen, 0, 1);
    try kolmogorov(sample, rv.uniform.probability, .{ 0, 1 });
}

test "weibull" {
    const slice = try all.alloc(f64, n);
    defer all.free(slice);
    const shapes = range(3);
    for (shapes) |shape| {
        const sample = rv.weibull.fill(slice, gen, shape, 1);
        try kolmogorov(sample, rv.weibull.probability, .{ shape, 1 });
    }
}
