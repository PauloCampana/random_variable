const std = @import("std");
const descriptive = @import("descriptive.zig");
const distribution = @import("distribution.zig");

pub const Htest = struct {
    name: []const u8,
    H0: []const u8,
    statistic: f64,
    quantile: f64,
    pvalue: f64,

    pub fn format(
        self: Htest,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}\n", .{self.name});
        try writer.print("H0: {s}\n", .{self.H0});
        try writer.print("statistic: {d:.3}\n", .{self.statistic});
        try writer.print("quantile: {d:.3}\n", .{self.quantile});
        try writer.print("pvalue: {d:.3}\n", .{self.pvalue});
    }
};

pub fn ztest(slice: []f64, mu0: f64, sd: f64, significance: f64) Htest {
    const len = @as(f64, @floatFromInt(slice.len));
    const xbar = descriptive.mean(slice);
    const statistic = (xbar - mu0) / sd * @sqrt(len);
    const quantile = distribution.quantile.normal(1 - significance / 2, mu0, sd);
    const pvalue = distribution.probability.normal(-@abs(statistic), mu0, sd) * 2;
    return Htest {
        .name = "One sample ztest",
        .H0 = "True mean is equal to mu0",
        .statistic = statistic,
        .quantile = quantile,
        .pvalue = pvalue,
    };
}

pub fn ttest(slice: []f64, mu0: f64, significance: f64) Htest {
    const len = @as(f64, @floatFromInt(slice.len));
    const xbar = descriptive.mean(slice);
    const se = descriptive.standardError(slice);
    const statistic = (xbar - mu0) / se;
    const quantile = distribution.quantile.t(1 - significance / 2, len - 1);
    const pvalue = distribution.probability.t(-@abs(statistic), len - 1);

    return Htest {
        .name = "One sample ttest",
        .H0 = "True mean is equal to mu0",
        .statistic = statistic,
        .quantile = quantile,
        .pvalue = pvalue,
    };
}
