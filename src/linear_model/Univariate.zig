//! ŷ = βX + ε

const std = @import("std");
const Matrix = @import("../Matrix.zig");
const common = @import("common.zig");
const Self = @This();

dependent: []f64,
independent: Matrix,
coefficient: []f64,
residue: []f64,
prediction: []f64,

pub fn fit(
    data: Matrix,
    dependent_index: usize,
    independent_indexes: []const usize,
    intercept: bool,
) !Self {
    const n = data.data[0].len;
    const p = independent_indexes.len + @intFromBool(intercept);

    const Y = try data.alloc(n, 1);
    defer Y.free();
    const dependent = try data.allocator.dupe(f64, Y.data[0]);
    errdefer data.allocator.free(dependent);
    @memcpy(Y.data[0], data.data[dependent_index]);

    const X = try data.alloc(n, p);
    errdefer X.free();
    if (intercept) {
        @memset(X.data[0], 1);
    }
    for (@intFromBool(intercept).., independent_indexes) |Xj, dataj| {
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
    defer B.free();
    const coefficient = try data.allocator.dupe(f64, B.data[0]);
    errdefer data.allocator.free(coefficient);

    const P = try X.multiplyMatrix(B);
    defer P.free();
    const prediction = try data.allocator.dupe(f64, P.data[0]);
    errdefer data.allocator.free(prediction);
    const E = try Y.dupe();
    defer E.free();
    E.subtractMatrix(P);
    const residue = try data.allocator.dupe(f64, E.data[0]);
    errdefer data.allocator.free(residue);

    return Self {
        .dependent = dependent,
        .independent = X,
        .coefficient = coefficient,
        .residue = residue,
        .prediction = prediction,
    };
}

pub fn free(self: Self) void {
    self.independent.allocator.free(self.dependent);
    self.independent.free();
    self.independent.allocator.free(self.coefficient);
    self.independent.allocator.free(self.residue);
    self.independent.allocator.free(self.prediction);
}

pub fn deviance(self: Self) f64 {
    return common.sse(self.residue);
}

pub fn rmse(self: Self) f64 {
    const mse = common.mse(self.residue, self.independent.data.len);
    return @sqrt(mse);
}

pub fn r2(self: Self, adjusted: bool) f64 {
    if (adjusted) {
        const mse = common.mse(self.residue, self.independent.data.len);
        const mst = common.mst(self.dependent);
        return 1 - mse / mst;
    } else {
        const sse = common.sse(self.residue);
        const sst = common.sst(self.dependent);
        return 1 - sse / sst;
    }
}

// pub fn ftest(self: Self, significance: f64) hypothesis.Htest {
//     const df1 = @as(f64, @floatFromInt(self.X.data.len));
//     const df2 = @as(f64, @floatFromInt(self.X.data[0].len)) - df1 - 1;
//     const statistic = blk: {
//         const mean = descriptive.mean.arithmetic(self.Y.data[0]);
//         var sum2reg: f64 = 0;
//         for (self.P.data[0]) |p| {
//             const d = p - mean;
//             sum2reg += d * d;
//         }
//         const sum2res = self.deviance();
//         const mean2reg = sum2reg / df1;
//         const mean2res = sum2res / df2;
//         break :blk mean2reg / mean2res;
//     };
//     const quantil = distribution.quantile.f(1 - significance, df1, df2);
//     const pvalue = 1 - distribution.probability.f(statistic, df1, df2);
//     return hypothesis.Htest {
//         .name = "F test for linear model adequacy",
//         .H0 = "All model coefficients are equal to 0",
//         .statistic = statistic,
//         .quantile = quantil,
//         .pvalue = pvalue,
//     };
// }

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




// test "linear_model.Univariate" {
//     const csv = @import("../csv.zig");
//     const diamonds = try csv.read(std.testing.allocator, "data/diamonds_numeric.csv", .{});
//     defer diamonds.free();

//     const model = try fit(diamonds, 3, &.{0, 1, 2, 4, 5, 6}, true);
//     defer model.free();
//     std.debug.print("{}", .{model.independent});
// }
