const std = @import("std");
const stdprob = @import("thirdyparty/prob.zig");
const RV = @import("RandomVariable.zig");

pub fn main() !void {
    var gen = std.rand.DefaultPrng
        .init(@intCast(std.time.nanoTimestamp()));
    var gpa = std.heap
        .GeneralPurposeAllocator(.{}) {};
    defer _ = gpa.deinit();
    var random = RV.random
        .setType(u64, f64)
        .init(gen.random(), gpa.allocator());

    const x = try random.normalSlice(1e6, 0, 1);
    defer random.allocator.free(x);
    const y = try random.normalSlice(1e6, 0, 3);
    defer random.allocator.free(y);

    std.debug.print("{d:.5}\n", .{RV.descriptive.sum(f64, x)});
    std.debug.print("{d:.5}\n", .{RV.descriptive.mean(f64, x)});
    std.debug.print("{d:.5}\n", .{RV.descriptive.variance(f64, x)});
    std.debug.print("{d:.5}\n", .{RV.descriptive.standardDeviation(f64, x)});
    std.debug.print("{d:.5}\n", .{RV.descriptive.standardError(f64, x)});
    std.debug.print("{d:.5}\n", .{RV.descriptive.skewness(f64, x)});
    std.debug.print("{d:.5}\n", .{RV.descriptive.kurtosis(f64, x)});
    std.debug.print("{d:.5}\n", .{RV.descriptive.covariance(f64, x, y)});
    std.debug.print("{d:.5}\n", .{RV.descriptive.correlation(f64, x, y)});
}
