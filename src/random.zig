const std = @import("std");
const lossyCast = std.math.lossyCast;

const assert = std.debug.assert;
const isFinite = std.math.isFinite; // tests false for both inf and nan

pub fn setType(comptime D: type, comptime C: type) type {
    if (C != f32 and C != f64) {
        @compileError("C must be f32 or f64, found " ++ @typeName(C));
    }
    return struct {
        const Self = @This();
        generator: std.rand.Random,
        allocator: std.mem.Allocator,

        pub fn init(generator: std.rand.Random, allocator: std.mem.Allocator) Self {
            return Self {
                .generator = generator,
                .allocator = allocator,
            };
        }

        /// Random number generator of Uniform distribution
        ///
        /// min and max ∈ (-∞,∞)
        pub fn uniform(self: Self, n: usize, min: C, max: C) ![]C {
            assert(isFinite(min) and isFinite(max));
            var slice = try self.allocator.alloc(C, n);
            if (min == max) {
                @memset(slice, min);
                return slice;
            }
            const amplitude = max - min;
            for (slice) |*x| {
                const uni = self.generator.float(C);
                x.* = min + amplitude * uni;
            }
            return slice;
        }

        /// Random number generator of Bernoulli distribution
        ///
        /// prob ∈ [0,1]
        pub fn bernoulli(self: Self, n: usize, prob: C) ![]D {
            assert(0 <= prob and prob <= 1);
            var slice = try self.allocator.alloc(D, n);
            if (prob == 0) {
                @memset(slice, 0);
                return slice;
            }
            if (prob == 1) {
                @memset(slice, 1);
                return slice;
            }
            for (slice) |*x| {
                const uni = self.generator.float(C);
                const ber = @intFromBool(uni < prob);
                x.* = lossyCast(D, ber);
            }
            return slice;
        }

        /// Random number generator of Geometric distribution
        ///
        /// prob ∈ (0,1]
        pub fn geometric(self: Self, n: usize, prob: C) ![]D {
            assert(0 < prob and prob <= 1);
            var slice = try self.allocator.alloc(D, n);
            if (prob == 1) {
                @memset(slice, 0);
                return slice;
            }
            const rate = -std.math.log1p(-prob);
            for (slice) |*x| {
                const exp = self.generator.floatExp(C);
                const geo = @trunc(exp / rate);
                x.* = lossyCast(D, geo);
            }
            return slice;
        }

        /// Random number generator of Poisson distribution
        ///
        /// lambda ∈ (0,∞)
        pub fn poisson(self: Self, n: usize, lambda: C) ![]D {
            assert(isFinite(lambda));
            assert(lambda > 0);
            var slice = try self.allocator.alloc(D, n);
            const p0 = @exp(-lambda);
            for (slice) |*x| {
                const uni = self.generator.float(C);
                var p = p0;
                var f = p0;
                var poi: C = 1;
                while (uni >= f) : (poi += 1) {
                    p *= lambda / poi;
                    f += p;
                }
                x.* = lossyCast(D, poi - 1);
            }
            return slice;
        }

        /// Random number generator of Binomial distribution
        ///
        /// size ∈ {0,1,2,⋯}
        ///
        /// prob ∈ [0,1]
        pub fn binomial(self: Self, n: usize, size: usize, prob: C) ![]D {
            assert(0 <= prob and prob <= 1);
            var slice = try self.allocator.alloc(D, n);
            if (size == 0 or prob == 0) {
                @memset(slice, 0);
                return slice;
            }
            if (prob == 1) {
                @memset(slice, lossyCast(D, size));
                return slice;
            }
            for (slice) |*x| {
                var bin: D = 0;
                for (0..size) |_| {
                    const uni = self.generator.float(C);
                    const ber = @intFromBool(uni < prob);
                    bin += lossyCast(D, ber);
                }
                x.* = bin;
            }
            return slice;
        }

        /// Random number generator of Negative Binomial distribution
        ///
        /// size ∈ {0,1,2,⋯}
        ///
        /// prob ∈ (0,1]
        pub fn negativeBinomial(self: Self, n: usize, size: usize, prob: C) ![]D {
            assert(0 < prob and prob <= 1);
            var slice = try self.allocator.alloc(D, n);
            if (prob == 1) {
                @memset(slice, 0);
                return slice;
            }
            const rate = -std.math.log1p(-prob);
            for (slice) |*x| {
                var nbi: D = 0;
                for (0..size) |_| {
                    const exp = self.generator.floatExp(C);
                    const geo = @trunc(exp / rate);
                    nbi += lossyCast(D, geo);
                }
                x.* = nbi;
            }
            return slice;
        }

        /// Random number generator of Exponential distribution
        ///
        /// rate ∈ (0,∞)
        pub fn exponential(self: Self, n: usize, rate: C) ![]C {
            assert(isFinite(rate));
            assert(rate > 0);
            var slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                const exp = self.generator.floatExp(C);
                x.* = exp / rate;
            }
            return slice;
        }

        /// Random number generator of Weibull distribution
        ///
        /// shape and rate ∈ (0,∞)
        pub fn weibull(self: Self, n: usize, shape: C, rate: C) ![]C {
            assert(isFinite(shape) and isFinite(rate));
            assert(shape > 0 and rate > 0);
            var slice = try self.allocator.alloc(C, n);
            const invshape = 1 / shape;
            for (slice) |*x| {
                const exp = self.generator.floatExp(C);
                const wei = std.math.pow(C, exp, invshape);
                x.* = wei / rate;
            }
            return slice;
        }

        /// Random number generator of Cauchy distribution
        ///
        /// location ∈ (-∞,∞)
        ///
        /// scale ∈ (0,∞)
        pub fn cauchy(self: Self, n: usize, location: C, scale: C) ![]C {
            assert(isFinite(location) and isFinite(scale));
            assert(scale > 0);
            var slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                const uni = self.generator.float(C);
                x.* = location + scale * @tan(std.math.pi * uni);
            }
            return slice;
        }

        /// Random number generator of Logistic distribution
        ///
        /// location ∈ (-∞,∞)
        ///
        /// scale ∈ (0,∞)
        pub fn logistic(self: Self, n: usize, location: C, scale: C) ![]C {
            assert(isFinite(location) and isFinite(scale));
            assert(scale > 0);
            var slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                const uni = self.generator.float(C);
                x.* = location + scale * @log(uni / (1 - uni));
            }
            return slice;
        }

        fn gammaFromCD(self: Self, c: C, d: C) C {
            while (true) {
                var v: C = undefined;
                var z: C = undefined;
                while (true) {
                    z = self.generator.floatNorm(C);
                    v = 1 + c * z;
                    if (v > 0) {
                        break;
                    }
                }
                v *= v * v;
                z *= z;
                const uni = self.generator.float(C);
                const cond0 = uni < 1 - 0.0331 * z * z;
                if (cond0 or @log(uni) < 0.5 * z + d * (1 - v + @log(v))) {
                    return d * v;
                }
            }
        }

        fn gammaSingle(self: Self, shape: C) C {
            const correct = shape >= 1;
            const d = blk: {
                const d0 = shape - 1.0 / 3.0;
                break :blk if (shape >= 1) d0 else d0 + 1;
            };
            const c = 1 / (3 * @sqrt(d));
            const gam = self.gammaFromCD(c, d);
            if (correct) {
                return gam;
            } else {
                const invshape = 1 / shape;
                const uni = self.generator.float(C);
                const correction = std.math.pow(C, uni, invshape);
                return gam * correction;
            }
        }

        /// Random number generator of Gamma distribution
        ///
        /// shape and rate ∈ (0,∞)
        pub fn gamma(self: Self, n: usize, shape: C, rate: C) ![]C {
            assert(isFinite(shape) and isFinite(rate));
            assert(shape > 0 and rate > 0);
            var slice = try self.allocator.alloc(C, n);
            const correct = shape >= 1;
            const d = blk: {
                const d0 = shape - 1.0 / 3.0;
                break :blk if (shape >= 1) d0 else d0 + 1;
            };
            const c = 1 / (3 * @sqrt(d));
            for (slice) |*x| {
                const gam = self.gammaFromCD(c, d);
                if (correct) {
                    x.* = gam / rate;
                } else {
                    const invshape = 1 / shape;
                    const uni = self.generator.float(C);
                    const correction = std.math.pow(C, uni, invshape);
                    x.* = gam / rate * correction;
                }
            }
            return slice;
        }

        /// Random number generator of Chi Squared distribution
        ///
        /// df ∈ (0,∞)
        pub fn chiSquared(self: Self, n: usize, df: C) ![]C {
            return self.gamma(n, 0.5 * df, 0.5);
        }

        /// Random number generator of C distribution
        ///
        /// df1 and df2 ∈ (0,∞)
        pub fn F(self: Self, n: usize, df1: C, df2: C) ![]C {
            assert(isFinite(df1) and isFinite(df2));
            assert(df1 > 0 and df2 > 0);
            var slice = try self.allocator.alloc(C, n);
            const ratio = df2 / df1;
            for (slice) |*x| {
                const chinum = self.gammaSingle(df1);
                const chiden = self.gammaSingle(df2);
                x.* = chinum / chiden * ratio;
            }
            return slice;
        }

        /// Random number generator of Beta distribution
        ///
        /// shape1 and shape2 ∈ (0,∞)
        pub fn beta(self: Self, n: usize, shape1: C, shape2: C) ![]C {
            assert(isFinite(shape1) and isFinite(shape2));
            assert(shape1 > 0 and shape2 > 0);
            var slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                const gam1 = self.gammaSingle(shape1);
                const gam2 = self.gammaSingle(shape2);
                x.* = gam1 / (gam1 + gam2);
            }
            return slice;
        }

        /// Random number generator of Normal distribution
        ///
        /// mean ∈ (-∞,∞)
        ///
        /// sd ∈ (0,∞)
        pub fn normal(self: Self, n: usize, mean: C, sd: C) ![]C {
            assert(isFinite(mean) and isFinite(sd));
            assert(sd > 0);
            var slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                const nor = self.generator.floatNorm(C);
                x.* = mean + sd * nor;
            }
            return slice;
        }

        /// Random number generator of Log-normal distribution
        ///
        /// meanlog ∈ (-∞,∞)
        ///
        /// sdlog ∈ (0,∞)
        pub fn logNormal(self: Self, n: usize, meanlog: C, sdlog: C) ![]C {
            assert(isFinite(meanlog) and isFinite(sdlog));
            assert(sdlog > 0);
            var slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                const nor = self.generator.floatNorm(C);
                const log = meanlog + sdlog * nor;
                x.* = @exp(log);
            }
            return slice;
        }

        /// Random number generator of t distribution
        ///
        /// df ∈ (0,∞)
        pub fn t(self: Self, n: usize, df: C) ![]C {
            assert(isFinite(df));
            assert(df > 0);
            if (df == 1) {
                return self.cauchy(n, 0, 1);
            }
            var slice = try self.allocator.alloc(C, n);
            for (slice) |*x| {
                const nor = self.generator.floatNorm(C);
                const chi = self.gammaSingle(0.5 * df) * 2;
                x.* = nor * @sqrt(df / chi);
            }
            return slice;
        }
    };
}

test {
    var allocator = std.testing.allocator;
    var gen = std.rand.DefaultPrng.init(0);
    const rv = setType(f64, f64).init(gen.random(), allocator);

    const actual = [_][]f64 {
        try rv.uniform         (10, 0, 1),
        try rv.bernoulli       (10, 0.2 ),
        try rv.geometric       (10, 0.2 ),
        try rv.poisson         (10, 3   ),
        try rv.binomial        (10, 10, 0.2),
        try rv.negativeBinomial(10, 10, 0.2),
        try rv.exponential     (10, 3   ),
        try rv.weibull         (10, 3, 5),
        try rv.cauchy          (10, 0, 1),
        try rv.logistic        (10, 0, 1),
        try rv.gamma           (10, 3, 5),
        try rv.chiSquared      (10, 3   ),
        try rv.F               (10, 3, 5),
        try rv.beta            (10, 3, 5),
        try rv.normal          (10, 0, 1),
        try rv.logNormal       (10, 0, 1),
        try rv.t               (10, 3   ),
    };

    defer {
        for (actual) |x| {
           rv.allocator.free(x);
        }
    }

    const expected = [_][10]f64 {
        [_]f64 {
            0x1.75d61490b23dfp-2, 0x1.a6f3dc380d507p-2,
            0x1.fdf91ec9a7bfcp-2, 0x1.ebf8c3bbe5e1ap-7,
            0x1.a04ebaf4a5eeap-2, 0x1.3c37757f08d9ap-6,
            0x1.490c75ab5026ep-1, 0x1.343e6464bc959p-1,
            0x1.da0a02389f0ffp-2, 0x1.0fc58c0424c16p-4,
        },
        [_]f64 {
            0x0.0000000000000p+0, 0x1.0000000000000p+0,
            0x1.0000000000000p+0, 0x1.0000000000000p+0,
            0x1.0000000000000p+0, 0x0.0000000000000p+0,
            0x0.0000000000000p+0, 0x1.0000000000000p+0,
            0x0.0000000000000p+0, 0x0.0000000000000p+0,
        },
        [_]f64 {
            0x0.0000000000000p+0, 0x0.0000000000000p+0,
            0x0.0000000000000p+0, 0x1.0000000000000p+0,
            0x1.8000000000000p+1, 0x1.c000000000000p+2,
            0x0.0000000000000p+0, 0x1.0000000000000p+1,
            0x1.0000000000000p+3, 0x0.0000000000000p+0,
        },
        [_]f64 {
            0x1.8000000000000p+1, 0x1.8000000000000p+1,
            0x1.0000000000000p+1, 0x1.0000000000000p+2,
            0x1.8000000000000p+2, 0x1.0000000000000p+1,
            0x1.0000000000000p+2, 0x1.0000000000000p+0,
            0x1.0000000000000p+2, 0x1.0000000000000p+0,
        },
        [_]f64 {
            0x1.0000000000000p+2, 0x1.0000000000000p+1,
            0x1.0000000000000p+0, 0x1.0000000000000p+0,
            0x1.0000000000000p+1, 0x1.0000000000000p+0,
            0x1.8000000000000p+1, 0x1.0000000000000p+2,
            0x1.0000000000000p+0, 0x1.8000000000000p+1,
        },
        [_]f64 {
            0x1.a000000000000p+4, 0x1.3800000000000p+5,
            0x1.6000000000000p+3, 0x1.1000000000000p+5,
            0x1.3000000000000p+4, 0x1.2000000000000p+6,
            0x1.c800000000000p+5, 0x1.6800000000000p+5,
            0x1.8000000000000p+4, 0x1.1000000000000p+5,
        },
        [_]f64 {
            0x1.8a3d82297d254p-1, 0x1.57faeaa068dbcp-3,
            0x1.365c0dee8f000p-1, 0x1.405b59be004a2p-5,
            0x1.4193dd540525fp-6, 0x1.c8af00dff3488p-1,
            0x1.10707d1369658p-2, 0x1.3e6d6f3209372p-1,
            0x1.257596a384a1dp-1, 0x1.a7345beb379e7p-7,
        },
        [_]f64 {
            0x1.17fd1079bee4dp-3, 0x1.093534721dd1ap-4,
            0x1.7b8d96630eb97p-3, 0x1.2023e33e970a3p-3,
            0x1.18af1f39f9be8p-3, 0x1.f694748699272p-3,
            0x1.8cb15fff97c3bp-3, 0x1.946eaeb959f87p-3,
            0x1.ef463d2a17b0ep-3, 0x1.6a8dee77ab5f2p-3,
        },
        [_]f64 {
           -0x1.f0f4755f6be93p-2, 0x1.5902e2247e2fcp+0,
            0x1.b11ce3e0c39a5p-1, 0x1.a37dd00b25ed4p+0,
           -0x1.e48a9f9b2b460p-2, 0x1.92992e7619486p+1,
            0x1.c07bb174e21c2p-1, 0x1.8a3fdcbbd4395p+0,
            0x1.14d5e506bb47ap+0, 0x1.8e34d291426fbp-5,
        } ,
        [_]f64 {
           -0x1.b63bf7c1cdbb8p+1,-0x1.f954c29562f69p-4,
           -0x1.6386c3fdaba55p-3,-0x1.cfdf04bb934b1p-1,
           -0x1.7a117a8cac893p+1, 0x1.fdabd963f5d8fp-1,
           -0x1.a13dac68352c6p-2,-0x1.5361dcbbe7544p+0,
            0x1.7f3101a49c8c0p+1, 0x1.4e01b98b3bfe1p+0,
        },
        [_]f64 {
            0x1.8f5d494006397p-1, 0x1.4b096cb0e45cdp-1,
            0x1.331537857b2d1p-1, 0x1.d02be70fe00b3p-1,
            0x1.5b62623c7d886p-2, 0x1.0dddcdfbe4180p-1,
            0x1.02a27cdfdbba8p-2, 0x1.028980742672ap+0,
            0x1.a9b5acc527f0dp-1, 0x1.b46a4fe86dfeap-2,
        },
        [_]f64 {
            0x1.e904f7e7fae4cp-2, 0x1.889daad60d316p+1,
            0x1.0ef7777e9843fp+0, 0x1.e9e7c405638e9p+1,
            0x1.131ad41d179c1p+1, 0x1.4767dc217447cp-2,
            0x1.811100c369fa4p+1, 0x1.279f08d85efabp+2,
            0x1.83222205f419ap+1, 0x1.e5bd10652c1dap+0,
        },
        [_]f64 {
            0x1.a29da1d59b052p+0, 0x1.03dd7132f85e0p+0,
            0x1.51e939634845dp-1, 0x1.0f19c4e24b089p+0,
            0x1.8a53d229ffd7fp+0, 0x1.2bd205bdce298p-1,
            0x1.b1774df5f70bfp-1, 0x1.14c23b10dd4b9p-1,
            0x1.2795cc2e7164cp+1, 0x1.4dc74a365ac19p+1,
        },
        [_]f64 {
            0x1.1c878a7f5bd89p-1, 0x1.944b36fbd162cp-2,
            0x1.20310d60a617cp-1, 0x1.65d5f38e1e303p-2,
            0x1.c54458579e6f0p-2, 0x1.143e46eaa5864p-1,
            0x1.aae4d6dfcb77cp-2, 0x1.f6405acad3430p-2,
            0x1.0f60c554c2386p-2, 0x1.3ae6b4a717686p-1,
        },
        [_]f64 {
            0x1.aabf3c2bee7e8p+0,-0x1.775360d9cbd85p-1,
           -0x1.66fbb93a4f575p+0, 0x1.09019d69e8a34p-1,
           -0x1.0f1f843a9ab0fp+1,-0x1.2b395530f5cb9p+0,
            0x1.b46626eaebb9ap+0, 0x1.5b9b79f475f48p-4,
           -0x1.ebdc36cf31f7bp-2, 0x1.78f3388119f54p-2,
        },
        [_]f64 {
            0x1.9d1b2d0b13c74p-1, 0x1.0f21e9e5a6ddcp-1,
            0x1.a986817288cdbp+2, 0x1.b573cb2474abcp+0,
            0x1.3ce0df4a271b1p+1, 0x1.5b4735d78783ep-1,
            0x1.412e0ecfaf630p-1, 0x1.f157d42ebe723p-1,
            0x1.a5b6389d77219p+0, 0x1.e822c099eba2ep-2,
        },
        [_]f64 {
            0x1.7726edf4c72afp-1, 0x1.2570804bf3e73p+0,
           -0x1.9947c4832202cp-1, 0x1.092304878fa13p+2,
            0x1.973cfa8449128p-5,-0x1.a65edf7b0f1aap-8,
            0x1.13588bf39ede7p-1,-0x1.2be3965711733p-1,
           -0x1.8125163350900p-1, 0x1.1f4ee9e5850c2p-1
        },
    };

    for (actual, expected) |act, exp| {
        for (act, exp) |a, e| {
            try std.testing.expectEqual(e, a);
        }
    }
}
