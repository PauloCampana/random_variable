//! Statistical functions

const std = @import("std");

pub const random = @import("random.zig");
pub const density = @import("density.zig");
pub const distribution = @import("distribution.zig");
pub const quantile = @import("quantile.zig");

pub const descriptive = @import("descriptive.zig");

pub const hypotesis = @import("hypotesis.zig");

test {
    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    defer _ = gpa.deinit();
    var gen = std.rand.DefaultPrng.init(0);
    const rv = random
        .setType(u64, f64)
        .init(gen.random(), gpa.allocator());

    const x = try rv.geometricSlice(100, 0.01);
    defer rv.allocator.free(x);

    std.debug.print("{d:.3}", .{x});

    std.testing.refAllDecls(@This());
}
