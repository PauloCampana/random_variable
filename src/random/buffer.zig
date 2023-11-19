const std = @import("std");
const implementation = @import("implementation.zig");

const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan

/// Fills a buffer with random variables.
///
/// Specify buffers of what types are returned from discrete and
/// continuous distributions respectively with D and C,
/// after that, use setGenerator.
///
/// first argument is always the buffer to be written
/// while the rest are the distribution's parameters.
pub fn Buffer(comptime D: type, comptime C: type) type {
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

        /// Specify an engine for random number generation
        pub fn setGenerator(generator: std.rand.Random) Self {
            return Self {.generator = generator};
        }

        /// min and max ∈ (-∞,∞)
        pub fn uniform(self: Self, buf: []C, min: C, max: C) []C {
            assert(isFinite(min) and isFinite(max));
            for (buf) |*x| {
                x.* = implementation.uniform(C, self.generator, min, max);
            }
            return buf;
        }

        /// prob ∈ [0,1]
        pub fn bernoulli(self: Self, buf: []D, prob: C) []D {
            assert(0 <= prob and prob <= 1);
            for (buf) |*x| {
                x.* = implementation.bernoulli(D, C, self.generator, prob);
            }
            return buf;
        }

        /// prob ∈ (0,1]
        pub fn geometric(self: Self, buf: []D, prob: C) []D {
            assert(0 < prob and prob <= 1);
            for (buf) |*x| {
                x.* = implementation.geometric(D, C, self.generator, prob);
            }
            return buf;
        }

        /// lambda ∈ (0,∞)
        pub fn poisson(self: Self, buf: []D, lambda: C) []D {
            assert(isFinite(lambda));
            assert(lambda > 0);
            for (buf) |*x| {
                x.* = implementation.poisson(D, C, self.generator, lambda);
            }
            return buf;
        }

        /// size ∈ {0,1,2,⋯}
        ///
        /// prob ∈ [0,1]
        pub fn binomial(self: Self, buf: []D, size: usize, prob: C) []D {
            assert(0 <= prob and prob <= 1);
            for (buf) |*x| {
                x.* =  implementation.binomial(D, C, self.generator, size, prob);
            }
            return buf;
        }

        /// size ∈ {1,2,3,⋯}
        ///
        /// prob ∈ (0,1]
        pub fn negativeBinomial(self: Self, buf: []D, size: usize, prob: C) []D {
            assert(0 < prob and prob <= 1);
            assert(size != 0);
            for (buf) |*x| {
                x.* = implementation.negativeBinomial(D, C, self.generator, size, prob);
            }
            return buf;
        }

        /// rate ∈ (0,∞)
        pub fn exponential(self: Self, buf: []C, rate: C) []C {
            assert(isFinite(rate));
            assert(rate > 0);
            for (buf) |*x| {
                x.* = implementation.exponential(C, self.generator, rate);
            }
            return buf;
        }

        /// shape and rate ∈ (0,∞)
        pub fn weibull(self: Self, buf: []C, shape: C, rate: C) []C {
            assert(isFinite(shape) and isFinite(rate));
            assert(shape > 0 and rate > 0);
            for (buf) |*x| {
                x.* = implementation.weibull(C, self.generator, shape, rate);
            }
            return buf;
        }

        /// location ∈ (-∞,∞)
        ///
        /// scale ∈ (0,∞)
        pub fn cauchy(self: Self, buf: []C, location: C, scale: C) []C {
            assert(isFinite(location) and isFinite(scale));
            assert(scale > 0);
            for (buf) |*x| {
                x.* = implementation.cauchy(C, self.generator, location, scale);
            }
            return buf;
        }

        /// location ∈ (-∞,∞)
        ///
        /// scale ∈ (0,∞)
        pub fn logistic(self: Self, buf: []C, location: C, scale: C) []C {
            assert(isFinite(location) and isFinite(scale));
            assert(scale > 0);
            for (buf) |*x| {
                x.* = implementation.logistic(C, self.generator, location, scale);
            }
            return buf;
        }

        /// shape and rate ∈ (0,∞)
        pub fn gamma(self: Self, buf: []C, shape: C, rate: C) []C {
            assert(isFinite(shape) and isFinite(rate));
            assert(shape > 0 and rate > 0);
            for (buf) |*x| {
                x.* = implementation.gamma(C, self.generator, shape, rate);
            }
            return buf;
        }

        /// df ∈ (0,∞)
        pub fn chiSquared(self: Self, buf: []C, df: C) []C {
            assert(isFinite(df));
            assert(df > 0);
            for (buf) |*x| {
                x.* = implementation.chiSquared(C, self.generator, df);
            }
            return buf;
        }

        /// df1 and df2 ∈ (0,∞)
        pub fn F(self: Self, buf: []C, df1: C, df2: C) []C {
            assert(isFinite(df1) and isFinite(df2));
            assert(df1 > 0 and df2 > 0);
            for (buf) |*x| {
                x.* = implementation.F(C, self.generator, df1, df2);
            }
            return buf;
        }

        /// shape1 and shape2 ∈ (0,∞)
        pub fn beta(self: Self, buf: []C, shape1: C, shape2: C) []C {
            assert(isFinite(shape1) and isFinite(shape2));
            assert(shape1 > 0 and shape2 > 0);
            for (buf) |*x| {
                x.* = implementation.beta(C, self.generator, shape1, shape2);
            }
            return buf;
        }

        /// mean ∈ (-∞,∞)
        ///
        /// sd ∈ (0,∞)
        pub fn normal(self: Self, buf: []C, mean: C, sd: C) []C {
            assert(isFinite(mean) and isFinite(sd));
            assert(sd > 0);
            for (buf) |*x| {
                x.* = implementation.normal(C, self.generator, mean, sd);
            }
            return buf;
        }

        /// meanlog ∈ (-∞,∞)
        ///
        /// sdlog ∈ (0,∞)
        pub fn logNormal(self: Self, buf: []C, meanlog: C, sdlog: C) []C {
            assert(isFinite(meanlog) and isFinite(sdlog));
            assert(sdlog > 0);
            for (buf) |*x| {
                x.* = implementation.logNormal(C, self.generator, meanlog, sdlog);
            }
            return buf;
        }

        /// df ∈ (0,∞)
        pub fn t(self: Self, buf: []C, df: C) []C {
            assert(isFinite(df));
            assert(df > 0);
            for (buf) |*x| {
                x.* = implementation.t(C, self.generator, df);
            }
            return buf;
        }
    };
}
