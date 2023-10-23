const std = @import("std");
const RandomVariable = @import("RandomVariable.zig");

pub fn main() !void {
    var gen = std.rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    var rv = RandomVariable.RandomVariable(u23, f64).init(gen.random());

    std.debug.print("{s}\n", .{@typeName(@TypeOf(rv.binomial(10,0.5)))});

    std.debug.print("\n{s:17}", .{"uniform:"});
    for (0..10) |_| {
        const x = rv.uniform(0, 1);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"bernoulli:"});
    for (0..10) |_| {
        const x = rv.bernoulli(0.5);
        std.debug.print("{:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"geometric:"});
    for (0..10) |_| {
        const x = rv.geometric(0.5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"poisson:"});
    for (0..10) |_| {
        const x = rv.poisson(5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"binomial:"});
    for (0..10) |_| {
        const x = rv.binomial(5, 0.5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"negativeBinomial:"});
    for (0..10) |_| {
        const x = rv.negativeBinomial(5, 0.5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"exponential:"});
    for (0..10) |_| {
        const x = rv.exponential(5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"weibull:"});
    for (0..10) |_| {
        const x = rv.weibull(5, 5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"cauchy:"});
    for (0..10) |_| {
        const x = rv.cauchy(0, 1);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"logistic:"});
    for (0..10) |_| {
        const x = rv.logistic(0, 1);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"gamma:"});
    for (0..10) |_| {
        const x = rv.gamma(5, 5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"chiSquared:"});
    for (0..10) |_| {
        const x = rv.chiSquared(5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"F:"});
    for (0..10) |_| {
        const x = rv.F(5, 5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"beta:"});
    for (0..10) |_| {
        const x = rv.beta(5, 5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"normal:"});
    for (0..10) |_| {
        const x = rv.normal(0, 1);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"logNormal:"});
    for (0..10) |_| {
        const x = rv.logNormal(0, 1);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n{s:17}", .{"t:"});
    for (0..10) |_| {
        const x = rv.t(5);
        std.debug.print("{d:10.3}", .{x});
    }
    std.debug.print("\n", .{});
}
