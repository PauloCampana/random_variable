const std = @import("std");
const rv = @import("root.zig");

const Htest = struct {
    name: []const u8,
    H0: []const u8,
    statistic: f64,
    quantile: f64,
    pvalue: f64,
};

pub fn print(t: Htest, writer: anytype) !void {
    try writer.print("{s}\n", .{t.name});
    try writer.print("H0: {s}\n", .{t.H0});
    try writer.print("statistic: {d:.3}\n", .{t.statistic});
    try writer.print("quantile: {d:.3}\n", .{t.quantile});
    try writer.print("pvalue: {d:.5}\n", .{t.pvalue});
}

pub fn ztest(slice: []f64, mu0: f64, sd: f64, significance: f64) !Htest {
    const len = @as(f64, @floatFromInt(slice.len));
    const xbar = rv.descriptive.mean(f64, slice);
    const statistic = (xbar - mu0) / sd * @sqrt(len);
    const quantile = try rv.quantile.normal(1 - significance / 2, mu0, sd);
    const pvalue = try rv.distribution.normal(-@abs(statistic), mu0, sd) * 2;
    return Htest {
        .name = "One sample ztest",
        .H0 = "True mean is equal to mu0",
        .statistic = statistic,
        .quantile = quantile,
        .pvalue = pvalue,
    };
}

pub fn ttest(slice: []f64, mu0: f64, significance: f64) !Htest {
    const len = @as(f64, @floatFromInt(slice.len));
    const xbar = rv.descriptive.mean(f64, slice);
    const se = rv.descriptive.standardError(f64, slice);
    const statistic = (xbar - mu0) / se;
    const quantile = try rv.quantile.t(1 - significance / 2, len - 1);
    const pvalue = try rv.distribution.t(-@abs(statistic), len - 1);

    return Htest {
        .name = "One sample ttest",
        .H0 = "True mean is equal to mu0",
        .statistic = statistic,
        .quantile = quantile,
        .pvalue = pvalue,
    };
}
