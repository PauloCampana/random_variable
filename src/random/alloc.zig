const std = @import("std");
const implementation = @import("implementation.zig");

const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan

/// Uses the allocator to provide a slice of random variables.
///
/// Specify slices of what types are returned from discrete and
/// continuous distributions respectively with D and C,
/// after that, use setGeneratorAllocator.
///
/// first argument is always the amount of variables to be generated
/// while the rest are the distribution's parameters.
pub fn Alloc(comptime D: type, comptime C: type) type {
    if (C != f32 and C != f64) {
        @compileError("C must be f32 or f64, found " ++ @typeName(C));
    }
    switch (@typeInfo(D)) {
        .Int, .Float => {},
        else => @compileError("D must be integer or float, found " ++ @typeName(D)),
    }
    return struct {
        const Self = @This();
        generator: std.rand.Random,
        allocator: std.mem.Allocator,

        /// Specify an engine for random number generation
        /// and an allocator for the returning slices
        pub fn init(generator: std.rand.Random, allocator: std.mem.Allocator) Self {
            return Self {
                .generator = generator,
                .allocator = allocator,
            };
        }

        /// min and max ∈ (-∞,∞)
        pub fn uniform(self: Self, n: usize, min: C, max: C) ![]C {
            assert(isFinite(min) and isFinite(max));
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.uniform(C, self.generator, min, max);
            }
            return slice;
        }

        /// prob ∈ [0,1]
        pub fn bernoulli(self: Self, n: usize, prob: C) ![]D {
            assert(0 <= prob and prob <= 1);
            const slice = try self.allocator.alloc(D, n);
            for (slice) |*x| {
                x.* = implementation.bernoulli(D, C, self.generator, prob);
            }
            return slice;
        }

        /// prob ∈ (0,1]
        pub fn geometric(self: Self, n: usize, prob: C) ![]D {
            assert(0 < prob and prob <= 1);
            const slice = try self.allocator.alloc(D, n);
            for (slice) |*x| {
                x.* = implementation.geometric(D, C, self.generator, prob);
            }
            return slice;
        }

        /// lambda ∈ (0,∞)
        pub fn poisson(self: Self, n: usize, lambda: C) ![]D {
            assert(isFinite(lambda));
            const slice = try self.allocator.alloc(D, n);
            for (slice) |*x| {
                x.* = implementation.poisson(D, C, self.generator, lambda);
            }
            return slice;
        }

        /// size ∈ {0,1,2,⋯}
        ///
        /// prob ∈ [0,1]
        pub fn binomial(self: Self, n: usize, size: usize, prob: C) ![]D {
            assert(0 <= prob and prob <= 1);
            const slice = try self.allocator.alloc(D, n);
            for (slice) |*x| {
                x.* =  implementation.binomial(D, C, self.generator, size, prob);
            }
            return slice;
        }

        /// size ∈ {1,2,3,⋯}
        ///
        /// prob ∈ (0,1]
        pub fn negativeBinomial(self: Self, n: usize, size: usize, prob: C) ![]D {
            assert(0 < prob and prob <= 1);
            assert(size != 0);
            const slice = try self.allocator.alloc(D, n);
            for (slice) |*x| {
                x.* = implementation.negativeBinomial(D, C, self.generator, size, prob);
            }
            return slice;
        }

        /// rate ∈ (0,∞)
        pub fn exponential(self: Self, n: usize, rate: C) ![]C {
            assert(isFinite(rate));
            assert(rate > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.exponential(C, self.generator, rate);
            }
            return slice;
        }

        /// shape and rate ∈ (0,∞)
        pub fn weibull(self: Self, n: usize, shape: C, rate: C) ![]C {
            assert(isFinite(shape) and isFinite(rate));
            assert(shape > 0 and rate > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.weibull(C, self.generator, shape, rate);
            }
            return slice;
        }

        /// location ∈ (-∞,∞)
        ///
        /// scale ∈ (0,∞)
        pub fn cauchy(self: Self, n: usize, location: C, scale: C) ![]C {
            assert(isFinite(location) and isFinite(scale));
            assert(scale > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.cauchy(C, self.generator, location, scale);
            }
            return slice;
        }

        /// location ∈ (-∞,∞)
        ///
        /// scale ∈ (0,∞)
        pub fn logistic(self: Self, n: usize, location: C, scale: C) ![]C {
            assert(isFinite(location) and isFinite(scale));
            assert(scale > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.logistic(C, self.generator, location, scale);
            }
            return slice;
        }

        /// shape and rate ∈ (0,∞)
        pub fn gamma(self: Self, n: usize, shape: C, rate: C) ![]C {
            assert(isFinite(shape) and isFinite(rate));
            assert(shape > 0 and rate > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.gamma(C, self.generator, shape, rate);
            }
            return slice;
        }

        /// df ∈ (0,∞)
        pub fn chiSquared(self: Self, n: usize, df: C) ![]C {
            assert(isFinite(df));
            assert(df > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.chiSquared(C, self.generator, df);
            }
            return slice;
        }

        /// df1 and df2 ∈ (0,∞)
        pub fn F(self: Self, n: usize, df1: C, df2: C) ![]C {
            assert(isFinite(df1) and isFinite(df2));
            assert(df1 > 0 and df2 > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.F(C, self.generator, df1, df2);
            }
            return slice;
        }

        /// shape1 and shape2 ∈ (0,∞)
        pub fn beta(self: Self, n: usize, shape1: C, shape2: C) ![]C {
            assert(isFinite(shape1) and isFinite(shape2));
            assert(shape1 > 0 and shape2 > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.beta(C, self.generator, shape1, shape2);
            }
            return slice;
        }

        /// mean ∈ (-∞,∞)
        ///
        /// sd ∈ (0,∞)
        pub fn normal(self: Self, n: usize, mean: C, sd: C) ![]C {
            assert(isFinite(mean) and isFinite(sd));
            assert(sd > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.normal(C, self.generator, mean, sd);
            }
            return slice;
        }

        /// meanlog ∈ (-∞,∞)
        ///
        /// sdlog ∈ (0,∞)
        pub fn logNormal(self: Self, n: usize, meanlog: C, sdlog: C) ![]C {
            assert(isFinite(meanlog) and isFinite(sdlog));
            assert(sdlog > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.logNormal(C, self.generator, meanlog, sdlog);
            }
            return slice;
        }

        /// df ∈ (0,∞)
        pub fn t(self: Self, n: usize, df: C) ![]C {
            assert(isFinite(df));
            assert(df > 0);
            const slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                x.* = implementation.t(C, self.generator, df);
            }
            return slice;
        }
    };
}
