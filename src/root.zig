//! Statistical functions

pub const random = @import("random.zig");
pub const density = @import("density.zig");
pub const distribution = @import("distribution.zig");
pub const quantile = @import("quantile.zig");

pub const descriptive = @import("descriptive.zig");
pub const hypotesis = @import("hypotesis.zig");

pub const Matrix = @import("Matrix.zig");

pub const csv = @import("csv.zig");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}

test density {
    _ = density.normal(3, 0, 1);
    _ = density.gamma(10, 3, 5);
    _ = density.binomial(5, 10, 0.2);
}

test distribution {
    _ = distribution.normal(3, 0, 1);
    _ = distribution.gamma(10, 3, 5);
    _ = distribution.binomial(5, 10, 0.2);
}

test quantile {
    _ = quantile.normal(0.95, 0, 1);
    _ = quantile.gamma(0.95, 3, 5);
    _ = quantile.binomial(0.95, 10, 0.2);
}
