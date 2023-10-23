const std = @import("std");
const RandomVariable = @import("RandomVariable.zig");

pub fn main() !void {
    const nano: u64 = @intCast(std.time.nanoTimestamp());
    var gen = std.rand.DefaultPrng.init(nano);
    var rv = RandomVariable {.generator = gen.random()};

    for (0..10) |_| {
        for (0..10) |_| {
            const x = rv.gamma(3,5);
            std.debug.print("{d:10.3}", .{x});
        }
        std.debug.print("\n", .{});
    }
}
