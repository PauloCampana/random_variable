const std = @import("std");
const implementation = @import("implementation.zig");

const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan

/// Generates only a single random variable.
///
/// Specify what types are returned from discrete and
/// continuous distributions respectively with D and C,
/// after that, use setGenerator.
///
/// function arguments are only the distribution's parameters.
pub fn Single(comptime D: type, comptime C: type) type {
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
        pub fn uniform(self: Self, min: C, max: C) C {
            assert(isFinite(min) and isFinite(max));
            return implementation.uniform(C, self.generator, min, max);
        }

        /// prob ∈ [0,1]
        pub fn bernoulli(self: Self, prob: C) D {
            assert(0 <= prob and prob <= 1);
            return implementation.bernoulli(D, C, self.generator, prob);
        }

        /// prob ∈ (0,1]
        pub fn geometric(self: Self, prob: C) D {
            assert(0 < prob and prob <= 1);
            return implementation.geometric(D, C, self.generator, prob);
        }

        /// lambda ∈ (0,∞)
        pub fn poisson(self: Self, lambda: C) D {
            assert(isFinite(lambda));
            assert(lambda > 0);
            return implementation.poisson(D, C, self.generator, lambda);
        }

        /// size ∈ {0,1,2,⋯}
        ///
        /// prob ∈ [0,1]
        pub fn binomial(self: Self, size: usize, prob: C) D {
            assert(0 <= prob and prob <= 1);
            return implementation.binomial(D, C, self.generator, size, prob);
        }

        /// size ∈ {1,2,3,⋯}
        ///
        /// prob ∈ (0,1]
        pub fn negativeBinomial(self: Self, size: usize, prob: C) D {
            assert(0 < prob and prob <= 1);
            assert(size != 0);
            return implementation.negativeBinomial(D, C, self.generator, size, prob);
        }

        /// rate ∈ (0,∞)
        pub fn exponential(self: Self, rate: C) C {
            assert(isFinite(rate));
            assert(rate > 0);
            return implementation.exponential(C, self.generator, rate);
        }

        /// shape and rate ∈ (0,∞)
        pub fn weibull(self: Self, shape: C, rate: C) C {
            assert(isFinite(shape) and isFinite(rate));
            assert(shape > 0 and rate > 0);
            return implementation.weibull(C, self.generator, shape, rate);
        }

        /// location ∈ (-∞,∞)
        ///
        /// scale ∈ (0,∞)
        pub fn cauchy(self: Self, location: C, scale: C) C {
            assert(isFinite(location) and isFinite(scale));
            assert(scale > 0);
            return implementation.cauchy(C, self.generator, location, scale);
        }

        /// location ∈ (-∞,∞)
        ///
        /// scale ∈ (0,∞)
        pub fn logistic(self: Self, location: C, scale: C) C {
            assert(isFinite(location) and isFinite(scale));
            assert(scale > 0);
            return implementation.logistic(C, self.generator, location, scale);
        }

        /// shape and rate ∈ (0,∞)
        pub fn gamma(self: Self, shape: C, rate: C) C {
            assert(isFinite(shape) and isFinite(rate));
            assert(shape > 0 and rate > 0);
            return implementation.gamma(C, self.generator, shape, rate);
        }

        /// df ∈ (0,∞)
        pub fn chiSquared(self: Self, df: C) C {
            assert(isFinite(df));
            assert(df > 0);
            return implementation.chiSquared(C, self.generator, df);
        }

        /// df1 and df2 ∈ (0,∞)
        pub fn F(self: Self, df1: C, df2: C) C {
            assert(isFinite(df1) and isFinite(df2));
            assert(df1 > 0 and df2 > 0);
            return implementation.F(C, self.generator, df1, df2);
        }

        /// shape1 and shape2 ∈ (0,∞)
        pub fn beta(self: Self, shape1: C, shape2: C) C {
            assert(isFinite(shape1) and isFinite(shape2));
            assert(shape1 > 0 and shape2 > 0);
            return implementation.beta(C, self.generator, shape1, shape2);
        }

        /// mean ∈ (-∞,∞)
        ///
        /// sd ∈ (0,∞)
        pub fn normal(self: Self, mean: C, sd: C) C {
            assert(isFinite(mean) and isFinite(sd));
            assert(sd > 0);
            return implementation.normal(C, self.generator, mean, sd);
        }

        /// meanlog ∈ (-∞,∞)
        ///
        /// sdlog ∈ (0,∞)
        pub fn logNormal(self: Self, meanlog: C, sdlog: C) C {
            assert(isFinite(meanlog) and isFinite(sdlog));
            assert(sdlog > 0);
            return implementation.logNormal(C, self.generator, meanlog, sdlog);
        }

        /// df ∈ (0,∞)
        pub fn t(self: Self, df: C) C {
            assert(isFinite(df));
            assert(df > 0);
            return implementation.t(C, self.generator, df);
        }
    };
}
