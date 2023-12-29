//! Random variable generation.

pub const single = @import("random/single.zig");
pub const buffer = @import("random/buffer.zig");
pub const alloc = @import("random/alloc.zig");

const std = @import("std");
test single {
    var prng = std.rand.DefaultPrng.init(0);
    const generator = prng.random();

    _ = single.normal(generator, 0, 1);
    _ = single.gamma(generator, 3, 5);
    _ = single.binomial(generator, 10, 0.2);
}

test buffer {
    var prng = std.rand.DefaultPrng.init(0);
    const generator = prng.random();
    var buf: [10]f64 = undefined;

    _ = buffer.normal(&buf, generator, 0, 1);
    _ = buffer.gamma(&buf, generator, 3, 5);
    _ = buffer.binomial(&buf, generator, 10, 0.2);
}

test alloc {
    var prng = std.rand.DefaultPrng.init(0);
    const generator = prng.random();

    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const nor = try alloc.normal(allocator, generator, 10, 0, 1);
    defer allocator.free(nor);
    const gam = try alloc.gamma(allocator, generator, 10, 3, 5);
    defer allocator.free(gam);
    const bin = try alloc.binomial(allocator, generator, 10, 10, 0.2);
    defer allocator.free(bin);
}
