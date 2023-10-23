const std = @import("std");

pub fn RandomVariable(comptime Tu: type, comptime Tf: type) type {
    return struct {
        const Self = @This();
        generator: std.rand.Random,

        pub fn init(r: std.rand.Random) Self {
            return Self {.generator = r};
        }

        pub fn uniform(self: Self, min: Tf, max: Tf) Tf {
            const uni = self.generator.float(Tf);
            return min + (max - min) * uni;
        }

        pub fn bernoulli(self: Self, prob: Tf) bool {
            const uni = self.generator.float(Tf);
            const ber = uni < prob;
            return ber;
        }

        pub fn geometric(self: Self, prob: Tf) Tu {
            const uni = self.generator.float(Tf);
            const geo = @floor(@log(uni) / std.math.log1p(-prob));
            return @intFromFloat(geo);
        }

        pub fn poisson(self: Self, lambda: Tf) Tu {
            const uni = self.generator.float(Tf);
            var p = @exp(-lambda);
            var f = p;
            var poi: Tu = 0;
            while (uni >= f) : (poi += 1) {
                p *= lambda / @as(Tf, @floatFromInt(poi + 1));
                f += p;
            }
            return poi;
        }

        pub fn binomial(self: Self, size: Tu, prob: Tf) Tu {
            var bin: Tu = 0;
            for (0..size) |_| {
                bin += @intFromBool(bernoulli(self, prob));
            }
            return bin;
        }

        pub fn negativeBinomial(self: Self, size: Tu, prob: Tf) Tu {
            var nbin: Tu = 0;
            for (0..size) |_| {
                nbin += geometric(self, prob);
            }
            return nbin;
        }

        pub fn exponential(self: Self, rate: Tf) Tf {
            const exp = self.generator.floatExp(Tf) / rate;
            return exp;
        }

        pub fn weibull(self: Self, shape: Tf, rate: Tf) Tf {
            const uni = self.generator.float(Tf);
            const wei = std.math.pow(Tf, -@log(uni), 1 / shape) / rate;
            return wei;
        }

        pub fn cauchy(self: Self, location: Tf, scale: Tf) Tf {
            const uni = self.generator.float(Tf);
            const cau = location + scale * @tan(std.math.pi * uni);
            return cau;
        }

        pub fn logistic(self: Self, location: Tf, scale: Tf) Tf {
            const uni = self.generator.float(Tf);
            const log = location + scale * @log(uni / (1 - uni));
            return log;
        }

        pub fn gamma(self: Self, shape: Tf, rate: Tf) Tf {
            const correct = shape >= 1;
            const d = blk: {
                const d = shape - 1.0 / 3.0;
                break: blk if (shape >= 1) d else d + 1;
            };
            const c = 1 / @sqrt(9 * d);

            var v: Tf = undefined;
            while (true) {
                var z: Tf = undefined;
                while (true) {
                    z = normal(self, 0, 1);
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
                    break;
                }
            }
            const gam = d * v;

            if (correct) {
                return gam / rate;
            } else {
                const uni = self.generator.float(Tf);
                const correction = std.math.pow(Tf, uni, 1 / shape);
                return gam * correction / rate;
            }
        }

        pub fn chiSquared(self: Self, df: Tf) Tf {
            return gamma(self, 0.5 * df, 0.5);
        }

        pub fn F(self: Self, df1: Tf, df2: Tf) Tf {
            const chi1 = chiSquared(self, df1);
            const chi2 = chiSquared(self, df2);
            const f = chi1 / chi2 / df1 * df2;
            return f;
        }

        pub fn beta(self: Self, shape1: Tf, shape2: Tf) Tf {
            const gam1 = gamma(self, shape1, 1);
            const gam2 = gamma(self, shape2, 1);
            const bet = gam1 / (gam1 + gam2);
            return bet;
        }

        pub fn normal(self: Self, mean: Tf, sd: Tf) Tf {
            const nor = self.generator.floatNorm(Tf);
            return mean + sd * nor;
        }

        pub fn logNormal(self: Self, meanlog: Tf, sdlog: Tf) Tf {
            return @exp(normal(self, meanlog, sdlog));
        }

        pub fn t(self: Self, df: Tf) Tf {
            const nor = normal(self, 0, 1);
            const chi = chiSquared(self, df);
            const T = nor * @sqrt(df / chi);
            return T;
        }
    };
}
