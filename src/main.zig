//! Statistical functions

const std = @import("std");

pub const random = @import("random.zig");
pub const density = @import("density.zig");
pub const distribution = @import("distribution.zig");
pub const quantile = @import("quantile.zig");

pub const descriptive = @import("descriptive.zig");

pub const hypotesis = @import("hypotesis.zig");

test {
    std.testing.refAllDecls(@This());
}
