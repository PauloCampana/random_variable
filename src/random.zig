//! Random variable generation
//! for common probability distributions
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe
//! such as ±NaN, ±Inf, probabilities outside [0,1],
//! negative or zero shape, df, rate or scale parameters.

pub const Single = @import("random/single.zig").Single;
pub const Buffer = @import("random/buffer.zig").Buffer;
pub const Alloc = @import("random/alloc.zig").Alloc;

const std = @import("std");

test Single {
    var prng = std.rand.DefaultPrng.init(0);
    const generator = prng.random();

    const random = Single(u64, f64)
        .setGenerator(generator);

    const nor = random.normal(0, 1);
    _ = nor;
}

test Buffer {
    var prng = std.rand.DefaultPrng.init(0);
    const generator = prng.random();

    const random = Buffer(u64, f64)
        .setGenerator(generator);

    var buf: [100]f64 = undefined;
    const gam = random.gamma(&buf, 3, 5);
    _ = gam;
}

test Alloc {
    var prng = std.rand.DefaultPrng.init(0);
    const generator = prng.random();

    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const random = Alloc(u64, f64)
        .setGeneratorAllocator(generator, allocator);

    const bin = try random.binomial(100, 10, 0.2);
    defer allocator.free(bin);
}
