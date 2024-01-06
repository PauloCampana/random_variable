//! Statistical functions

pub const distribution = @import("distribution.zig");

pub const descriptive = @import("descriptive.zig");
pub const hypotesis = @import("hypothesis.zig");

pub const Matrix = @import("Matrix.zig");
pub const csv = @import("csv.zig");
pub const linear_model = @import("linear_model.zig");

const std = @import("std");
test {
    std.testing.refAllDeclsRecursive(@This());
}

test descriptive {
    const carat = &[_]f64 {0.23, 0.21, 0.23, 0.29, 0.31};
    const price = &[_]f64 {326, 326, 327, 334, 335};
    _ = descriptive.mean.arithmetic(carat);
    _ = descriptive.variance(price);
    _ = descriptive.correlation.kendall(carat, price);
}

test Matrix {
    const X = try Matrix
        .init(std.testing.allocator)
        .createFromSliceOfSlices(&.{
            &.{0.23, 0.21, 0.23, 0.29, 0.31},
            &.{326, 326, 327, 334, 335},
        });
    defer X.free();
    const XT = try X.transpose();
    defer XT.free();
    const XTX = try XT.multiplyMatrix(X);
    defer XTX.free();
    const XTXinv = try XTX.inverse();
    defer XTXinv.free();
}
