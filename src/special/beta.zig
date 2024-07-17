const std = @import("std");
const normal = @import("normal.zig");

const exp_underflow = -745.1332191019412076235;
const exp_overflow = 709.782712893383973096;
const gamma_overflow = 171.624376956302725;
const small = std.math.floatEps(f64);
const big = 1 / small;

pub fn probability(xx: f64, aa: f64, bb: f64) f64 {
    if (xx <= 0) {
        return 0;
    }
    if (xx >= 1) {
        return 1;
    }

    if (bb * xx <= 1 and xx <= 0.95) {
        return pseries(xx, aa, bb);
    }

    const flipped = xx > aa / (aa + bb);
    const a, const b, const x, const xc = if (flipped) .{
        bb, aa, 1 - xx, xx,
    } else .{
        aa, bb, xx, 1 - xx,
    };

    if (flipped and b * x <= 1 and x <= 0.95) {
        return 1 - pseries(x, a, b);
    }

    const w = if (x * (a + b - 2) < a - 1)
        incbcf(x, a, b)
    else
        incbd(x, a, b) / xc;

    const log_xa = a * @log(x);
    const log_xcb = b * @log(xc);
    if (a + b < gamma_overflow and @abs(log_xa) < exp_overflow and @abs(log_xcb) < exp_overflow) {
        const xa = std.math.pow(f64, x, a);
        const xcb = std.math.pow(f64, xc, b);
        const inv_beta = std.math.gamma(f64, a + b) / (std.math.gamma(f64, a) * std.math.gamma(f64, b));
        const t = xa * xcb * w / a * inv_beta;
        return if (flipped) 1 - t else t;
    } else {
        const log_beta = std.math.lgamma(f64, a) + std.math.lgamma(f64, b) - std.math.lgamma(f64, a + b);
        const log = log_xa + log_xcb + @log(w / a) - log_beta;
        return if (flipped) 1 - @exp(log) else @exp(log);
    }
}

fn pseries(x: f64, a: f64, b: f64) f64 {
    const z = small / a;

    var n: f64 = 1;
    var num: f64 = 1;
    var term = 1 / a;
    var sum = term;
    while (@abs(term) > z) : (n += 1) {
        num *= (n - b) * x / n;
        term = num / (a + n);
        sum += term;
    }

    const log_xa = a * @log(x);
    if (a + b < gamma_overflow and @abs(log_xa) < exp_overflow) {
        const inv_beta = std.math.gamma(f64, a + b) / (std.math.gamma(f64, a) * std.math.gamma(f64, b));
        const xa = std.math.pow(f64, x, a);
        return xa * inv_beta * sum;
    } else {
        const log_beta = std.math.lgamma(f64, a) + std.math.lgamma(f64, b) - std.math.lgamma(f64, a + b);
        const log = log_xa - log_beta + @log(sum);
        return @exp(log);
    }
}

fn incbcf(x: f64, a: f64, b: f64) f64 {
    var k1 = a;
    var k2 = a + b;
    var k3 = a;
    var k4 = a + 1;
    var k5: f64 = 1;
    var k6 = b - 1;
    var k7 = k4;
    var k8 = a + 2;

    var num_km2: f64 = 0;
    var den_km2: f64 = 1;
    var num_km1: f64 = 1;
    var den_km1: f64 = 1;
    var frac: f64 = 1;

    for (0..300) |_| {
        const x1_k = -(x * k1 * k2) / (k3 * k4);
        const num1_k = num_km1 + num_km2 * x1_k;
        const den1_k = den_km1 + den_km2 * x1_k;
        num_km2 = num_km1;
        den_km2 = den_km1;
        num_km1 = num1_k;
        den_km1 = den1_k;

        const x2_k = (x * k5 * k6) / (k7 * k8);
        const num_k = num_km1 + num_km2 * x2_k;
        const den_k = den_km1 + den_km2 * x2_k;
        num_km2 = num_km1;
        den_km2 = den_km1;
        num_km1 = num_k;
        den_km1 = den_k;

        if (den_k != 0) {
            const r = num_k / den_k;
            if (r == frac) {
                break;
            }
            frac = r;
        }

        k1 += 1;
        k2 += 1;
        k3 += 2;
        k4 += 2;
        k5 += 1;
        k6 -= 1;
        k7 += 2;
        k8 += 2;

        if (@abs(num_k) + @abs(den_k) > big) {
            num_km2 *= small;
            den_km2 *= small;
            num_km1 *= small;
            den_km1 *= small;
        }
        if (@abs(num_k) < small or @abs(den_k) < small) {
            num_km2 *= big;
            den_km2 *= big;
            num_km1 *= big;
            den_km1 *= big;
        }
    }
    return frac;
}

fn incbd(x: f64, a: f64, b: f64) f64 {
    const z = x / (1 - x);

    var k1 = a;
    var k2 = b - 1;
    var k3 = a;
    var k4 = a + 1;
    var k5: f64 = 1;
    var k6 = a + b;
    var k7 = k4;
    var k8 = a + 2;

    var num_km2: f64 = 0;
    var den_km2: f64 = 1;
    var num_km1: f64 = 1;
    var den_km1: f64 = 1;
    var frac: f64 = 1;

    for (0..300) |_| {
        const x1_k = -(z * k1 * k2) / (k3 * k4);
        const num1_k = num_km1 + num_km2 * x1_k;
        const den1_k = den_km1 + den_km2 * x1_k;
        num_km2 = num_km1;
        den_km2 = den_km1;
        num_km1 = num1_k;
        den_km1 = den1_k;

        const x2_k = (z * k5 * k6) / (k7 * k8);
        const num_k = num_km1 + num_km2 * x2_k;
        const den_k = den_km1 + den_km2 * x2_k;
        num_km2 = num_km1;
        den_km2 = den_km1;
        num_km1 = num_k;
        den_km1 = den_k;

        if (den_k != 0) {
            const r = num_k / den_k;
            if (r == frac) {
                break;
            }
            frac = r;
        }

        k1 += 1;
        k2 -= 1;
        k3 += 2;
        k4 += 2;
        k5 += 1;
        k6 += 1;
        k7 += 2;
        k8 += 2;

        if (@abs(num_k) + @abs(den_k) > big) {
            num_km2 *= small;
            den_km2 *= small;
            num_km1 *= small;
            den_km1 *= small;
        }
        if (@abs(num_k) < small or @abs(den_k) < small) {
            num_km2 *= big;
            den_km2 *= big;
            num_km1 *= big;
            den_km1 *= big;
        }
    }
    return frac;
}

pub fn quantile(yy0: f64, aa: f64, bb: f64) f64 {
    if (yy0 <= 0) {
        return 0;
    }
    if (yy0 >= 1) {
        return 1;
    }

    const State = enum {
        ihalve,
        newton,
    };

    var state: State, var rflg, var dithresh: f64, var a, var b, var x, var y, var y0 = blk: {
        if (aa <= 1 or bb <= 1) {
            const x = aa / (aa + bb);
            const y = probability(x, aa, bb);
            break :blk .{ .ihalve, false, 1e-6, aa, bb, x, y, yy0 };
        }

        const rflg, const a, const b, const y0, const yp = if (yy0 > 0.5) .{
            true, bb, aa, 1 - yy0, normal.quantile(yy0),
        } else .{
            false, aa, bb, yy0, -normal.quantile(yy0),
        };

        const lgm = (yp * yp - 3) / 6;
        const x = 2 / (1 / (2 * a - 1) + 1 / (2 * b - 1));
        const d = 2 * (yp * @sqrt(x + lgm) / x - (1 / (2 * b - 1) - 1 / (2 * a - 1)) * (lgm + 5 / 6 - 2 / (3 * x)));
        if (d < exp_underflow) {
            return done(rflg, 0);
        }
        const newx = a / (a + b * @exp(d));
        const y = probability(newx, a, b);
        const state: State = if (@abs((y - y0) / y0) < 0.2) .newton else .ihalve;
        break :blk .{ state, rflg, 1e-4, a, b, newx, y, y0 };
    };

    var x0: f64 = 0;
    var yl: f64 = 0;
    var x1: f64 = 1;
    var yh: f64 = 1;
    var nflg = false;

    outer: while (true) switch (state) {
        .ihalve => {
            var dir: f64 = 0;
            var di: f64 = 0.5;

            for (0..100) |i| {
                if (i != 0) {
                    x = x0 + di * (x1 - x0);
                    if (x == 1) {
                        x = 1 - small;
                    }
                    if (x == 0) {
                        di = 0.5;
                        x = x0 + di * (x1 - x0);
                        if (x == 0) {
                            return done(rflg, 0);
                        }
                    }
                    y = probability(x, a, b);
                    if (@abs((x1 - x0) / (x1 + x0)) < dithresh) {
                        state = .newton;
                        continue :outer;
                    }
                    if (@abs((y - y0) / y0) < dithresh) {
                        state = .newton;
                        continue :outer;
                    }
                }
                if (y < y0) {
                    x0 = x;
                    yl = y;
                    if (dir < 0) {
                        dir = 0;
                        di = 0.5;
                    } else if (dir > 3) {
                        di = 1 - (1 - di) * (1 - di);
                    } else if (dir > 1) {
                        di = 0.5 * di + 0.5;
                    } else {
                        di = (y0 - y) / (yh - yl);
                    }

                    dir += 1;
                    if (x0 > 0.75) {
                        if (rflg) {
                            rflg = false;
                            a = aa;
                            b = bb;
                            y0 = yy0;
                        } else {
                            rflg = true;
                            a = bb;
                            b = aa;
                            y0 = 1 - yy0;
                        }
                        x = 1 - x;
                        y = probability(x, a, b);
                        x0 = 0;
                        yl = 0;
                        x1 = 1;
                        yh = 1;
                        state = .ihalve;
                        continue :outer;
                    }
                } else {
                    x1 = x;
                    if (rflg and x1 < small) {
                        return done(rflg, 0);
                    }
                    yh = y;
                    if (dir > 0) {
                        dir = 0;
                        di = 0.5;
                    } else if (dir < -3) {
                        di = di * di;
                    } else if (dir < -1) {
                        di = 0.5 * di;
                    } else {
                        di = (y - y0) / (yh - yl);
                    }
                    dir -= 1;
                }
            }

            if (x0 >= 1) {
                x = 1 - small;
                return done(rflg, x);
            }
            if (x <= 0) {
                return done(rflg, 0);
            }

            state = .newton;
        },

        .newton => {
            if (nflg) {
                return done(rflg, x);
            }

            nflg = true;
            const lgm = std.math.lgamma(f64, a + b) - std.math.lgamma(f64, a) - std.math.lgamma(f64, b);

            for (0..8) |i| {
                if (i != 0) {
                    y = probability(x, a, b);
                }
                if (y < yl) {
                    x = x0;
                    y = yl;
                } else if (y > yh) {
                    x = x1;
                    y = yh;
                } else if (y < y0) {
                    x0 = x;
                    yl = y;
                } else {
                    x1 = x;
                    yh = y;
                }

                if (x == 1 or x == 0) {
                    break;
                }

                var d = (a - 1) * @log(x) + (b - 1) * @log(1 - x) + lgm;
                if (d < exp_underflow) {
                    return done(rflg, x);
                }
                if (d > exp_overflow) {
                    break;
                }
                d = @exp(d);
                d = (y - y0) / d;
                var xt = x - d;
                if (xt <= x0) {
                    y = (x - x0) / (x1 - x0);
                    xt = x0 + 0.5 * y * (x - x0);
                    if (xt <= 0) {
                        break;
                    }
                }
                if (xt >= x1) {
                    y = (x1 - x) / (x1 - x0);
                    xt = x1 - 0.5 * y * (x1 - x);
                    if (xt >= 1) {
                        break;
                    }
                }
                x = xt;
                if (@abs(d / x) < 128 * small) {
                    return done(rflg, x);
                }
            }

            dithresh = 256 * small;
            state = .ihalve;
        },
    };
    unreachable;
}

fn done(flipped: bool, t: f64) f64 {
    if (flipped) {
        if (t <= small) {
            return 1 - small;
        } else {
            return 1 - t;
        }
    }
    return t;
}
