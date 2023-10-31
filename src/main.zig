const std = @import("std");
const RandomVariable = @import("RandomVariable.zig");

pub fn main() !void {
    var gen = std.rand.DefaultPrng
        .init(@intCast(std.time.nanoTimestamp()));
    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    var rv = RandomVariable
        .setType(u64, f64)
        .init(gen.random(), gpa.allocator());

    try testrv(rv);
}

pub fn testrv(rv: anytype) !void {
    std.debug.print("uni {d:10.3}\n", .{try rv.uniformSlice(10, 0, 1)});
    std.debug.print("ber {any:10}\n", .{try rv.bernoulliSlice(10, 0.5)});
    std.debug.print("geo {d:10.3}\n", .{try rv.geometricSlice(10, 0.5)});
    std.debug.print("poi {d:10.3}\n", .{try rv.poissonSlice(10, 5)});
    std.debug.print("bin {d:10.3}\n", .{try rv.binomialSlice(10, 10, 0.5)});
    std.debug.print("nbi {d:10.3}\n", .{try rv.negativeBinomialSlice(10, 10, 0.5)});
    std.debug.print("exp {d:10.3}\n", .{try rv.exponentialSlice(10, 0.5)});
    std.debug.print("wei {d:10.3}\n", .{try rv.weibullSlice(10, 5, 0.5)});
    std.debug.print("cau {d:10.3}\n", .{try rv.cauchySlice(10, 0, 1)});
    std.debug.print("log {d:10.3}\n", .{try rv.logisticSlice(10, 0, 1)});
    std.debug.print("gam {d:10.3}\n", .{try rv.gammaSlice(10, 5, 5)});
    std.debug.print("chi {d:10.3}\n", .{try rv.chiSquaredSlice(10, 5)});
    std.debug.print("F   {d:10.3}\n", .{try rv.FSlice(10, 5, 5)});
    std.debug.print("bet {d:10.3}\n", .{try rv.betaSlice(10, 5, 5)});
    std.debug.print("nor {d:10.3}\n", .{try rv.normalSlice(10, 0, 1)});
    std.debug.print("lno {d:10.3}\n", .{try rv.logNormalSlice(10, 0, 1)});
    std.debug.print("t   {d:10.3}\n", .{try rv.tSlice(10, 5)});
}
