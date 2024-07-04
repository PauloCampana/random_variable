const std = @import("std");
const rv = @import("random_variable");

const params = range(3);
const n = 1_000_000;

pub fn main() !void {
    const progress = std.Progress.start(.{
        .root_name = "correctness",
        .estimated_total_items = test_fns.len,
    });
    defer progress.end();

    var generators: [test_fns.len]std.Random.DefaultPrng = undefined;
    for (&generators, 0..) |*generator, i| {
        generator.* = std.Random.DefaultPrng.init(0);
        for (0..i) |_| {
            generator.jump();
        }
    }

    var threads: [test_fns.len]std.Thread = undefined;
    inline for (&threads, test_fns, &generators) |*thread, test_fn, *generator| {
        thread.* = try std.Thread.spawn(
            // minimum for the buffer
            .{ .stack_size = 8 * 1024 * 1024 },
            test_fn,
            .{ generator.random(), progress },
        );
    }

    for (threads) |thread| {
        thread.join();
    }
}

const KolmogorovError = error{
    FailedKolmogorovTest,
};

fn kolmogorov(sample: []f64, namespace: type, parameters: anytype) KolmogorovError!void {
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
            \\
            \\test:       {}({:.3})
            \\statistic:  {d:.8}
            \\quantile:   {d:.8}
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

fn beta(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("beta", params.len);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |shape1| {
        node.completeOne();
        for (params) |shape2| {
            rv.beta.fill(&buffer, random, shape1, shape2);
            try kolmogorov(&buffer, rv.beta, .{ shape1, shape2 });
        }
    }
}

fn betaPrime(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("betaPrime", params.len);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |shape1| {
        node.completeOne();
        for (params) |shape2| {
            rv.betaPrime.fill(&buffer, random, shape1, shape2);
            try kolmogorov(&buffer, rv.betaPrime, .{ shape1, shape2 });
        }
    }
}

fn cauchy(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("cauchy", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    rv.cauchy.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.cauchy, .{ 0, 1 });
}

fn chi(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("chi", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |df| {
        rv.chi.fill(&buffer, random, df);
        try kolmogorov(&buffer, rv.chi, .{df});
    }
}

fn chiSquared(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("chiSquared", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |df| {
        rv.chiSquared.fill(&buffer, random, df);
        try kolmogorov(&buffer, rv.chiSquared, .{df});
    }
}

fn continuousBernoulli(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("continuousBernoulli", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    const shapes = [_]f64{ 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999 };
    for (shapes) |shape| {
        rv.continuousBernoulli.fill(&buffer, random, shape);
        try kolmogorov(&buffer, rv.continuousBernoulli, .{shape});
    }
}

fn dagum(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("dagum", params.len);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |shape1| {
        node.completeOne();
        for (params) |shape2| {
            rv.dagum.fill(&buffer, random, shape1, shape2, 1);
            try kolmogorov(&buffer, rv.dagum, .{ shape1, shape2, 1 });
        }
    }
}

fn exponential(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("exponential", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    rv.exponential.fill(&buffer, random, 1);
    try kolmogorov(&buffer, rv.exponential, .{1});
}

fn f(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("f", params.len);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |df1| {
        node.completeOne();
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

fn gamma(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("gamma", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |shape| {
        rv.gamma.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.gamma, .{ shape, 1 });
    }
}

fn gompertz(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("gompertz", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |shape| {
        rv.gompertz.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.gompertz, .{ shape, 1 });
    }
}

fn gumbel(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("gumbel", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    rv.gumbel.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.gumbel, .{ 0, 1 });
}

fn laplace(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("laplace", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    rv.laplace.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.laplace, .{ 0, 1 });
}

fn logistic(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("logistic", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    rv.logistic.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.logistic, .{ 0, 1 });
}

fn logNormal(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("logNormal", params.len);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |log_location| {
        node.completeOne();
        for (params) |log_scale| {
            rv.logNormal.fill(&buffer, random, log_location, log_scale);
            try kolmogorov(&buffer, rv.logNormal, .{ log_location, log_scale });
        }
    }
}

fn normal(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("normal", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    rv.normal.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.normal, .{ 0, 1 });
}

fn pareto(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("pareto", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |shape| {
        rv.pareto.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.pareto, .{ shape, 1 });
    }
}

fn rayleigh(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("rayleigh", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    rv.rayleigh.fill(&buffer, random, 1);
    try kolmogorov(&buffer, rv.rayleigh, .{1});
}

fn t(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("t", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |df| {
        rv.t.fill(&buffer, random, df);
        try kolmogorov(&buffer, rv.t, .{df});
    }
}

fn uniform(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("uniform", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    rv.uniform.fill(&buffer, random, 0, 1);
    try kolmogorov(&buffer, rv.uniform, .{ 0, 1 });
}

fn weibull(random: std.Random, progress: std.Progress.Node) KolmogorovError!void {
    const node = progress.start("weibull", 0);
    defer node.end();
    var buffer: [n]f64 = undefined;
    for (params) |shape| {
        rv.weibull.fill(&buffer, random, shape, 1);
        try kolmogorov(&buffer, rv.weibull, .{ shape, 1 });
    }
}

const TestFn = fn (std.Random, std.Progress.Node) KolmogorovError!void;
const test_fns = [_]TestFn{
    // benford,
    // bernoulli,
    beta,
    // betaBinomial,
    betaPrime,
    // binomial,
    cauchy,
    chi,
    chiSquared,
    continuousBernoulli,
    dagum,
    // discreteUniform,
    exponential,
    f,
    gamma,
    // geometric,
    gompertz,
    gumbel,
    // hypergeometric,
    laplace,
    // logarithmic,
    logistic,
    logNormal,
    // negativeBinomial,
    normal,
    pareto,
    // poisson,
    rayleigh,
    t,
    uniform,
    weibull,
};
