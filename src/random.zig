const std = @import("std");
const isFinite = std.math.isFinite;

pub fn setType(comptime Tu: type, comptime Tf: type) type {
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

        pub fn uniformSingle(self: Self, min: Tf, max: Tf) Tf {
            const uni = self.generator.float(Tf);
            return min + (max - min) * uni;
        }
        pub fn uniformSlice(self: Self, n: usize, min: Tf, max: Tf) ![]Tf {
            if (!isFinite(min) or !isFinite(max)) {
                return error.NonFiniteParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            if (min == max) {
                @memset(slice, min);
                return slice;
            }
            const amplitude = max - min;
            for (slice) |*x| {
                const uni = self.generator.float(Tf);
                x.* = min + amplitude * uni;
            }
            return slice;
        }

        pub fn bernoulliSingle(self: Self, prob: Tf) bool {
            const uni = self.generator.float(Tf);
            return uni < prob;
        }
        pub fn bernoulliSlice(self: Self, n: usize, prob: Tf) ![]bool {
            if (!isFinite(prob)) {
                return error.NonFiniteParam;
            }
            if (prob < 0 or prob > 1) {
                return error.ProbOutside01;
            }
            var slice = try self.allocator.alloc(bool, n);
            if (prob == 0) {
                @memset(slice, false);
                return slice;
            }
            if (prob == 1) {
                @memset(slice, true);
                return slice;
            }
            for (slice) |*x| {
                const uni = self.generator.float(Tf);
                x.* = uni < prob;
            }
            return slice;
        }

        pub fn geometricSingle(self: Self, prob: Tf) Tu {
            const rate = -std.math.log1p(-prob);
            const exp = self.generator.floatExp(Tf);
            return @intFromFloat(exp / rate);
        }
        pub fn geometricSlice(self: Self, n: usize, prob: Tf) ![]Tu {
            if (!isFinite(prob)) {
                return error.NonFiniteParam;
            }
            if (prob < 0 or prob > 1) {
                return error.ProbOutside01;
            }
            if (prob == 0) {
                return error.ZeroParam;
            }
            var slice = try self.allocator.alloc(Tu, n);
            if (prob == 1) {
                @memset(slice, 0);
                return slice;
            }
            const rate = -std.math.log1p(-prob);
            for (slice) |*x| {
                const exp = self.generator.floatExp(Tf);
                x.* = @intFromFloat(exp / rate);
            }
            return slice;
        }

        pub fn poissonSingle(self: Self, lambda: Tf) Tu {
            const uni = self.generator.float(Tf);
            var p = @exp(-lambda);
            var f = p;
            var poi = 0;
            while (uni >= f) : (poi += 1) {
                p *= lambda / @as(Tf, @floatFromInt(poi + 1));
                f += p;
            }
            return poi;
        }
        pub fn poissonSlice(self: Self, n: usize, lambda: Tf) ![]Tu {
            if (!isFinite(lambda)) {
                return error.NonFiniteParam;
            }
            if (lambda < 0) {
                return error.NegativeParam;
            }
            var slice = try self.allocator.alloc(Tu, n);
            if (lambda == 0) {
                @memset(slice, 0);
                return slice;
            }
            const p0 = @exp(-lambda);
            for (slice) |*x| {
                const uni = self.generator.float(Tf);
                var p = p0;
                var f = p;
                x.* = 0;
                while (uni >= f) : (x.* += 1) {
                    p *= lambda / @as(Tf, @floatFromInt(x.* + 1));
                    f += p;
                }
            }
            return slice;
        }

        pub fn binomialSingle(self: Self, size: Tu, prob: Tf) Tu {
            var bin = 0;
            for (0..size) |_| {
                const uni = self.generator.float(Tf);
                bin += @intFromBool(uni < prob);
            }
            return bin;
        }
        pub fn binomialSlice(self: Self, n: usize, size: Tu, prob: Tf) ![]Tu {
            if (!isFinite(prob)) {
                return error.NonFiniteParam;
            }
            if (prob < 0 or prob > 1) {
                return error.ProbOutside01;
            }
            var slice = try self.allocator.alloc(Tu, n);
            if (size == 0 or prob == 0) {
                @memset(slice, 0);
                return slice;
            }
            if (prob == 1) {
                @memset(slice, size);
                return slice;
            }
            for (slice) |*x| {
                x.* = 0;
                for (0..size) |_| {
                    const uni = self.generator.float(Tf);
                    x.* += @intFromBool(uni < prob);
                }
            }
            return slice;
        }

        pub fn negativeBinomial(self: Self, size: Tu, prob: Tf) Tu {
            const rate = -std.math.log1p(-prob);
            var nbin = 0;
            for (0..size) |_| {
                const exp = self.generator.floatExp(Tf);
                nbin += @intFromFloat(exp / rate);
            }
            return nbin;
        }
        pub fn negativeBinomialSlice(self: Self, n: usize, size: Tu, prob: Tf) ![]Tu {
            if (!isFinite(prob)) {
                return error.NonFiniteParam;
            }
            if (prob < 0 or prob > 1) {
                return error.ProbOutside01;
            }
            if (size == 0) {
                return error.ZeroParam;
            }
            var slice = try self.allocator.alloc(Tu, n);
            if (prob == 1) {
                @memset(slice, 0);
                return slice;
            }
            const rate = -std.math.log1p(-prob);
            for (slice) |*x| {
                x.* = 0;
                for (0..size) |_| {
                    const exp = self.generator.floatExp(Tf);
                    x.* += @intFromFloat(exp / rate);
                }
            }
            return slice;
        }

        pub fn exponentialSingle(self: Self, rate: Tf) Tf {
            const exp = self.generator.floatExp(Tf);
            return exp / rate;
        }
        pub fn exponentialSlice(self: Self, n: usize, rate: Tf) ![]Tf {
            if (!isFinite(rate)) {
                return error.NonFiniteParam;
            }
            if (rate < 0) {
                return error.NegativeParam;
            }
            if (rate == 0) {
                return error.ZeroParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            for (slice) |*x| {
                const exp = self.generator.floatExp(Tf);
                x.* = exp / rate;
            }
            return slice;
        }

        pub fn weibullSingle(self: Self, shape: Tf, rate: Tf) Tf {
            const invshape = 1 / shape;
            const exp = self.generator.floatExp(Tf);
            const wei = std.math.pow(Tf, exp, invshape);
            return wei / rate;
        }
        pub fn weibullSlice(self: Self, n: usize, shape: Tf, rate: Tf) ![]Tf {
            if (!isFinite(shape) or !isFinite(rate)) {
                return error.NonFiniteParam;
            }
            if (shape < 0 or rate < 0) {
                return error.NegativeParam;
            }
            if (shape == 0 or shape == 0) {
                return error.ZeroParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            const invshape = 1 / shape;
            for (slice) |*x| {
                const exp = self.generator.floatExp(Tf);
                const wei = std.math.pow(Tf, exp, invshape);
                x.* = wei / rate;
            }
            return slice;
        }

        pub fn cauchySingle(self: Self, location: Tf, scale: Tf) Tf {
            const uni = self.generator.float(Tf);
            return location + scale * @tan(std.math.pi * uni);
        }
        pub fn cauchySlice(self: Self, n: usize, location: Tf, scale: Tf) ![]Tf {
            if (!isFinite(location) or !isFinite(scale)) {
                return error.NonFiniteParam;
            }
            if (scale < 0) {
                return error.NegativeParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            if (scale == 0) {
                @memset(slice, location);
                return slice;
            }
            for (slice) |*x| {
                const uni = self.generator.float(Tf);
                x.* = location + scale * @tan(std.math.pi * uni);
            }
            return slice;
        }

        pub fn logisticSingle(self: Self, location: Tf, scale: Tf) Tf {
            const uni = self.generator.float(Tf);
            return location + scale * @log(uni / (1 - uni));
        }
        pub fn logisticSlice(self: Self, n: usize, location: Tf, scale: Tf) ![]Tf {
            if (!isFinite(location) or !isFinite(scale)) {
                return error.NonFiniteParam;
            }
            if (scale < 0) {
                return error.NegativeParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            if (scale == 0) {
                @memset(slice, location);
                return slice;
            }
            for (slice) |*x| {
                const uni = self.generator.float(Tf);
                x.* = location + scale * @log(uni / (1 - uni));
            }
            return slice;
        }

        fn gammaFromCD(self: Self, c: Tf, d: Tf) Tf {
            while (true) {
                var v: f64 = undefined;
                var z: f64 = undefined;
                while (true) {
                    z = self.generator.floatNorm(Tf);
                    v = 1 + c * z;
                    if (v > 0) {
                        break;
                    }
                }
                v *= v * v;
                z *= z;
                const uni = self.generator.float(Tf);
                if (
                    uni < 1 - 0.0331 * z * z or
                    @log(uni) < 0.5 * z + d * (1 - v + @log(v))
                ) {
                    return d * v;
                }
            }
        }
        pub fn gammaSingle(self: Self, shape: Tf, rate: Tf) Tf {
            const correct = shape >= 1;
            const d = blk: {
                const d = shape - 1.0 / 3.0;
                break :blk if (correct) d else d + 1;
            };
            const c = 1 / @sqrt(9 * d);
            if (correct) {
                const gam = self.gammaFromCD(c, d);
                return gam / rate;
            } else {
                const invshape = 1 / shape;
                const uni = self.generator.float(Tf);
                const correction = std.math.pow(Tf, uni, invshape);
                const gam = self.gammaFromCD(c, d);
                return gam / rate * correction;
            }
        }
        pub fn gammaSlice(self: Self, n: usize, shape: Tf, rate: Tf) ![]Tf {
            if (!isFinite(shape) or !isFinite(rate)) {
                return error.NonFiniteParam;
            }
            if (shape < 0 or rate < 0) {
                return error.NegativeParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            if (shape == 0 or rate == 0) {
                @memset(slice, 0);
                return slice;
            }
            const correct = shape >= 1;
            const d = blk: {
                const d = shape - 1.0 / 3.0;
                break :blk if (shape >= 1) d else d + 1;
            };
            const c = 1 / @sqrt(9 * d);
            if (correct) {
                for (slice) |*x| {
                    const gam = self.gammaFromCD(c, d);
                    x.* = gam / rate;
                }
            } else {
                const invshape = 1 / shape;
                for (slice) |*x| {
                    const uni = self.generator.float(Tf);
                    const correction = std.math.pow(Tf, uni, invshape);
                    const gam = self.gammaFromCD(c, d);
                    x.* = gam / rate * correction;
                }
            }
            return slice;
        }

        pub fn chiSquaredSingle(self: Self, df: Tf) Tf {
            return self.gammaSingle(0.5 * df, 0.5);
        }
        pub fn chiSquaredSlice(self: Self, n: usize, df: Tf) ![]Tf {
            return self.gammaSlice(n, 0.5 * df, 0.5);
        }

        pub fn FSingle(self: Self, df1: Tf, df2: Tf) Tf {
            const ratio = df2 / df1;
            const chi1 = self.chiSquaredSingle(df1);
            const chi2 = self.chiSquaredSingle(df2);
            return chi1 / chi2 * ratio;
        }
        pub fn FSlice(self: Self, n: usize, df1: Tf, df2: Tf) ![]Tf {
            if (!isFinite(df1) or !isFinite(df2)) {
                return error.NonFiniteParam;
            }
            if (df1 < 0 or df2 < 0) {
                return error.NegativeParam;
            }
            if (df1 == 0 or df2 == 0) {
                return error.ZeroParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            const ratio = df2 / df1;
            for (slice) |*x| {
                const chi1 = self.chiSquaredSingle(df1);
                const chi2 = self.chiSquaredSingle(df2);
                x.* = chi1 / chi2 * ratio;
            }
            return slice;
        }

        pub fn betaSingle(self: Self, shape1: Tf, shape2: Tf) Tf {
            const gam1 = self.gammaSingle(shape1, 1);
            const gam2 = self.gammaSingle(shape2, 1);
            return gam1 / (gam1 + gam2);
        }
        pub fn betaSlice(self: Self, n: usize, shape1: Tf, shape2: Tf) ![]Tf {
            if (!isFinite(shape1) or !isFinite(shape2)) {
                return error.NonFiniteParam;
            }
            if (shape1 < 0 or shape2 < 0) {
                return error.NegativeParam;
            }
            if (shape1 == 0 or shape2 == 0) {
                return error.ZeroParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            for (slice) |*x| {
                const gam1 = self.gammaSingle(shape1, 1);
                const gam2 = self.gammaSingle(shape2, 1);
                x.* = gam1 / (gam1 + gam2);
            }
            return slice;
        }

        pub fn normalSingle(self: Self, mean: Tf, sd: Tf) Tf {
            const nor = self.generator.floatNorm(Tf);
            return mean + sd * nor;
        }
        pub fn normalSlice(self: Self, n: usize, mean: Tf, sd: Tf) ![]Tf {
            if (!isFinite(mean) or !isFinite(sd)) {
                return error.NonFiniteParam;
            }
            if (sd < 0) {
                return error.NegativeParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            if (sd == 0) {
                @memset(slice, mean);
                return slice;
            }
            for (slice) |*x| {
                const nor = self.generator.floatNorm(Tf);
                x.* = mean + sd * nor;
            }
            return slice;
        }

        pub fn logNormalSingle(self: Self, meanlog: Tf, sdlog: Tf) Tf {
            const nor = self.generator.floatNorm(Tf);
            return @exp(meanlog + sdlog * nor);
        }
        pub fn logNormalSlice(self: Self, n: usize, meanlog: Tf, sdlog: Tf) ![]Tf {
            if (!isFinite(meanlog) or !isFinite(sdlog)) {
                return error.NonFiniteParam;
            }
            if (sdlog < 0) {
                return error.NegativeParam;
            }
            var slice = try self.allocator.alloc(Tf, n);
            if (sdlog == 0) {
                @memset(slice, @exp(meanlog));
                return slice;
            }
            for (slice) |*x| {
                const nor = self.generator.floatNorm(Tf);
                x.* = @exp(meanlog + sdlog * nor);
            }
            return slice;
        }

        pub fn tSingle(self: Self, df: Tf) Tf {
            const nor = self.generator.floatNorm(Tf);
            const chi = self.chiSquaredSingle(df);
            const t = nor * @sqrt(df / chi);
            return t;
        }
        pub fn tSlice(self: Self, n: usize, df: Tf) ![]Tf {
            if (!isFinite(df)) {
                return error.NonFiniteParam;
            }
            if (df < 0) {
                return error.NegativeParam;
            }
            if (df == 0) {
                return error.ZeroParam;
            }
            if (df == 1) {
                return self.cauchySlice(n, 0, 1);
            }
            var slice = try self.allocator.alloc(Tf, n);
            for (slice) |*x| {
                const nor = self.generator.floatNorm(Tf);
                const chi = self.chiSquaredSingle(df);
                x.* = nor * @sqrt(df / chi);
            }
            return slice;
        }
    };
}
