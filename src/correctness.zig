const std = @import("std");
const rv = @import("random_variable");

const n = 1_000_000;
const sizes = [_]u64{ 0, 1, 2, 5, 10, 20 };
const integers = [_]i64{ -10, -3, -1, 0, 1, 3, 10 };
const probs = [_]f64{ 0, 0.01, 0.1, 0.5, 0.9, 0.99, 1 };
const shapes = [_]f64{
    std.math.pow(f64, std.math.sqrt2, -3),
    std.math.pow(f64, std.math.sqrt2, -2),
    std.math.pow(f64, std.math.sqrt2, -1),
    std.math.pow(f64, std.math.sqrt2, 0),
    std.math.pow(f64, std.math.sqrt2, 1),
    std.math.pow(f64, std.math.sqrt2, 2),
    std.math.pow(f64, std.math.sqrt2, 3),
};

fn kolmogorov(
    sample: []f64,
    namespace: type,
    parameters: anytype,
) !void {
    const len: f64 = @floatFromInt(sample.len);
    std.mem.sortUnstable(f64, sample, {}, std.sort.asc(f64));

    var index: f64 = 1;
    var statistic: f64 = 0;
    for (sample) |q| {
        const empirical = index / len;
        const theoretical = @call(.auto, namespace.probability, .{q} ++ parameters);
        statistic = @max(statistic, @abs(empirical - theoretical));
        index += 1;
    }

    const quantile = 1.94947 / @sqrt(len); // alpha = 0.001
    if (statistic > quantile) {
        std.log.err(
            \\     {}{:.3}
            \\statistic:  {d:.8}
            \\quantile:   {d:.8}
            \\
        , .{ namespace, parameters, statistic, quantile });
        return error.FailedKolmogorovTest;
    }
}

fn pearson(
    sample: []const f64,
    namespace: type,
    parameters: anytype,
) !void {
    const len: f64 = @floatFromInt(sample.len);
    var map = std.AutoHashMap(i64, f64).init(std.testing.allocator);
    defer map.deinit();

    for (sample) |x| {
        const query = try map.getOrPut(@intFromFloat(x));
        if (query.found_existing) {
            query.value_ptr.* += 1;
        } else {
            query.value_ptr.* = 1;
        }
    }

    var statistic: f64 = -len;
    var map_it = map.iterator();
    while (map_it.next()) |entry| {
        const x: f64 = @floatFromInt(entry.key_ptr.*);
        const density = @call(.auto, namespace.density, .{x} ++ parameters);
        const expected = len * density;
        const observed = entry.value_ptr.*;
        statistic += observed * observed / expected;
    }

    const df = @max(map.count(), 2) - 1;
    const pvalue = rv.chiSquared.survival(statistic, @floatFromInt(df));
    if (pvalue < 0.001) {
        std.log.err(
            \\     {}{:.3}
            \\statistic: {d:.8}
            \\pvalue:    {d:.8}
            \\
        , .{ namespace, parameters, statistic, pvalue });
        return error.FailedPearsonTest;
    }
}

test "benford" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (sizes[2..]) |size| { // size >= 2
        rv.benford.fill(&buffer, random, size);
        try pearson(&buffer, rv.benford, .{size});
    }
}

test "bernoulli" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (probs) |prob| {
        rv.bernoulli.fill(&buffer, random, prob);
        try pearson(&buffer, rv.bernoulli, .{prob});
    }
}

test "beta" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (shapes) |shape1| {
        for (shapes) |shape2| {
            rv.beta.fill(&buffer, random, shape1, shape2);
            try kolmogorov(&buffer, rv.beta, .{ shape1, shape2 });
        }
    }
}

test "betaBinomial" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (sizes) |size| {
        for (shapes) |shape1| {
            for (shapes) |shape2| {
                rv.betaBinomial.fill(&buffer, random, size, shape1, shape2);
                try pearson(&buffer, rv.betaBinomial, .{ size, shape1, shape2 });
            }
        }
    }
}

test "betaPrime" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (shapes) |shape1| {
        for (shapes) |shape2| {
            rv.betaPrime.fill(&buffer, random, shape1, shape2);
            try kolmogorov(&buffer, rv.betaPrime, .{ shape1, shape2 });
        }
    }
}

test "binomial" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (sizes) |size| {
        for (probs) |prob| {
            rv.binomial.fill(&buffer, random, size, prob);
            try pearson(&buffer, rv.binomial, .{ size, prob });
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
    for (shapes) |df| {
        rv.chi.fill(&buffer, random, df);
        try kolmogorov(&buffer, rv.chi, .{df});
    }
}

test "chiSquared" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (shapes) |df| {
        rv.chiSquared.fill(&buffer, random, df);
        try kolmogorov(&buffer, rv.chiSquared, .{df});
    }
}

test "continuousBernoulli" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (probs[1..5]) |shape| { // prob > 0 and prob < 1
        rv.continuousBernoulli.fill(&buffer, random, shape);
        try kolmogorov(&buffer, rv.continuousBernoulli, .{shape});
    }
}

test "dagum" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (shapes) |shape1| {
        for (shapes) |shape2| {
            rv.dagum.fill(&buffer, random, shape1, shape2, 1);
            try kolmogorov(&buffer, rv.dagum, .{ shape1, shape2, 1 });
        }
    }
}

test "discreteUniform" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (integers) |min| {
        for (integers) |max| {
            if (min > max) continue; // skip invalid parameters
            rv.discreteUniform.fill(&buffer, random, min, max);
            try pearson(&buffer, rv.discreteUniform, .{ min, max });
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
    for (shapes) |df1| {
        for (shapes) |df2| {
            rv.f.fill(&buffer, random, df1, df2);
            try kolmogorov(&buffer, rv.f, .{ df1, df2 });
        }
    }
}

test "gamma" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (shapes) |shape| {
        rv.gamma.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.gamma, .{ shape, 1 });
    }
}

test "geometric" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    // HACK: prob == 0.01 generates way too many unique numbers
    for (probs[2..]) |prob| { // prob > 0
        rv.geometric.fill(&buffer, random, prob);
        try pearson(&buffer, rv.geometric, .{prob});
    }
}

test "gompertz" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (shapes) |shape| {
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

test "hypergeometric" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (sizes) |total| {
        for (sizes) |good| {
            for (sizes) |tries| {
                if (good > total or tries > total) continue; // skip invalid parameters
                rv.hypergeometric.fill(&buffer, random, total, good, tries);
                try pearson(&buffer, rv.hypergeometric, .{ total, good, tries });
            }
        }
    }
}

test "laplace" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    rv.laplace.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.laplace, .{ 0, 1 });
}

test "logarithmic" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    // HACK: prob == 0.99 generates way too many unique numbers
    for (probs[1..5]) |prob| { // prob > 0 and prob < 1
        rv.logarithmic.fill(&buffer, random, prob);
        try pearson(&buffer, rv.logarithmic, .{prob});
    }
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
    for (shapes) |log_location| {
        for (shapes) |log_scale| {
            rv.logNormal.fill(&buffer, random, log_location, log_scale);
            try kolmogorov(&buffer, rv.logNormal, .{ log_location, log_scale });
        }
    }
}

test "negativeBinomial" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (sizes[1..]) |size| { // size > 0
        // HACK: prob == 0.01 and 0.1 generates way too many unique numbers
        for (probs[3..]) |prob| { // prob > 0
            rv.negativeBinomial.fill(&buffer, random, size, prob);
            try pearson(&buffer, rv.negativeBinomial, .{ size, prob });
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
    for (shapes) |shape| {
        rv.pareto.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.pareto, .{ shape, 1 });
    }
}

test "poisson" {
    var buffer: [n]f64 = undefined;
    var gen = std.Random.DefaultPrng.init(0);
    const random = gen.random();
    for (shapes) |shape| {
        rv.poisson.fill(&buffer, random, shape);
        try pearson(&buffer, rv.poisson, .{shape});
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
    for (shapes) |df| {
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
    for (shapes) |shape| {
        rv.weibull.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.weibull, .{ shape, 1 });
    }
}
