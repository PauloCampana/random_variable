//! Y = XB + E

// TODO split @This() into two: univariate and multivariate
const std = @import("std");
const Matrix = @import("Matrix.zig");
const descriptive = @import("descriptive.zig");
const hypothesis = @import("hypothesis.zig");
const distribution = @import("distribution.zig");
const Self = @This();

Y: Matrix,
X: Matrix,
B: Matrix,
E: Matrix,
P: Matrix,

pub fn fit(
    data: Matrix,
    dependent: []const usize,
    independent: []const usize,
    intercept: bool,
) !Self {
    const n = data.data[0].len;
    const q = dependent.len;
    const p = if (intercept) independent.len + 1 else independent.len;

    const Y = try data.alloc(n, q);
    errdefer Y.free();
    for (Y.data, dependent) |*col, j| {
        @memcpy(col.*, data.data[j]);
    }

    const X = try data.alloc(n, p);
    errdefer X.free();
    if (intercept) {
        @memset(X.data[0], 1);
    }
    for (independent, @intFromBool(intercept)..) |dataj, Xj| {
        @memcpy(X.data[Xj], data.data[dataj]);
    }

    const XT = try X.transpose();
    defer XT.free();
    const XTX = try XT.multiplyMatrix(X);
    defer XTX.free();
    const XTXinv = try XTX.inverse();
    defer XTXinv.free();
    const XTXinvXT = try XTXinv.multiplyMatrix(XT);
    defer XTXinvXT.free();
    const B = try XTXinvXT.multiplyMatrix(Y);
    errdefer B.free();

    const P = try X.multiplyMatrix(B);
    errdefer P.free();
    const E = try Y.dupe();
    errdefer E.free();
    E.subtractMatrix(P);

    return Self {
        .Y = Y,
        .X = X,
        .B = B,
        .E = E,
        .P = P,
    };
}

pub fn free(self: Self) void {
    self.Y.free();
    self.X.free();
    self.B.free();
    self.E.free();
    self.P.free();
}

pub fn deviance(self: Self) f64 {
    var sum2: f64 = 0;
    for (self.E.data[0]) |e| {
        sum2 += e * e;
    }
    return sum2;
}

pub fn rmse(self: Self) f64 {
    const df = self.X.data[0].len - self.X.data.len;
    const mse = self.deviance() / @as(f64, @floatFromInt(df));
    return @sqrt(mse);
}

pub fn r2(self: Self, adjusted: bool) f64 {
    const mean = descriptive.mean.arithmetic(self.Y.data[0]);
    var sum2tot: f64 = 0;
    for (self.Y.data[0]) |y| {
        const d = y - mean;
        sum2tot += d * d;
    }
    const sum2res = self.deviance();
    if (adjusted) {
        const dfres = self.X.data[0].len - self.X.data.len;
        const mean2res = sum2res / @as(f64, @floatFromInt(dfres));
        const dftot = self.X.data[0].len - 1;
        const mean2tot = sum2tot / @as(f64, @floatFromInt(dftot));
        return 1 - mean2res / mean2tot;
    }
    return 1 - sum2res / sum2tot;
}

pub fn ftest(self: Self, significance: f64) hypothesis.Htest {
    const df1 = @as(f64, @floatFromInt(self.X.data.len));
    const df2 = @as(f64, @floatFromInt(self.X.data[0].len)) - df1 - 1;
    const statistic = blk: {
        const mean = descriptive.mean.arithmetic(self.Y.data[0]);
        var sum2reg: f64 = 0;
        for (self.P.data[0]) |p| {
            const d = p - mean;
            sum2reg += d * d;
        }
        const sum2res = self.deviance();
        const mean2reg = sum2reg / df1;
        const mean2res = sum2res / df2;
        break :blk mean2reg / mean2res;
    };
    const quantil = distribution.quantile.f(1 - significance, df1, df2);
    const pvalue = 1 - distribution.probability.f(statistic, df1, df2);
    return hypothesis.Htest {
        .name = "F test for linear model adequacy",
        .H0 = "All model coefficients are equal to 0",
        .statistic = statistic,
        .quantile = quantil,
        .pvalue = pvalue,
    };
}

pub fn coefficients(self: Self) []f64 {
    return self.B.data[0];
}

// support multivariate regression on
// deviance, rmse, r2

// summary
// coef
// effects
// residuals
//     regular
//     standard
//     t
// fitted
// vcov
// predict
// confint
// influence
// fit

// f statistic
// beta cov matrix

// anova

test "LinearModel" {
    // const csv = @import("csv.zig");
    // const diamonds = try csv.read(std.testing.allocator, "data/diamonds_numeric.csv", .{});
    // defer diamonds.free();

    // const model = try fit(diamonds, &.{3}, &.{0,1,2,4,5,6}, true);
    // defer model.free();
    // std.debug.print("{}", .{model});
    // std.debug.print("deviance = {d}\n", .{model.deviance()});
    // std.debug.print("rmse = {d}\n", .{model.rmse()});
    // std.debug.print("r2 = {d}\n", .{model.r2(false)});
    // std.debug.print("adj r2 = {d}\n", .{model.r2(true)});
    // std.debug.print("coefficients = {d}\n", .{model.coefficients()});
    // std.debug.print("ftest = {}\n", .{try model.ftest(0.05)});
}
