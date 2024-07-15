const std = @import("std");
const normal = @import("normal.zig");

const exp_underflow = -745.1332191019412076235;
const small = std.math.floatEps(f64);
const big = 1 / small;
const max_float = std.math.floatMax(f64);

pub fn probability(x: f64, a: f64) f64 {
    if (x <= 0) {
        return 0;
    }
    if (x == std.math.inf(f64)) {
        return 1;
    }
    if (x > 1 and x > a) {
        return 1 - survival(x, a);
    }

    const log_factor = a * @log(x) - x - std.math.lgamma(f64, a);
    if (log_factor < exp_underflow) {
        return 0;
    }
    const factor = @exp(log_factor);

    var den = a;
    var term: f64 = 1;
    var sum: f64 = 1;
    var prev_sum: f64 = 0;
    while (sum > prev_sum) {
        den += 1;
        term *= x / den;
        prev_sum = sum;
        sum += term;
    }

    return factor * sum / a;
}

pub fn survival(x: f64, a: f64) f64 {
    if (x <= 0) {
        return 1;
    }
    if (x == std.math.inf(f64)) {
        return 0;
    }
    if (x < 1 or x < a) {
        return 1 - probability(x, a);
    }

    const log_factor = a * @log(x) - x - std.math.lgamma(f64, a);
    if (log_factor < exp_underflow) {
        return 0;
    }
    const factor = @exp(log_factor);

    var y = 1 - a;
    var z = x + y + 1;
    var c: f64 = 0;
    var num_km2: f64 = 1;
    var den_km2 = x;
    var num_km1 = x + 1;
    var den_km1 = z * x;
    var frac = num_km1 / den_km1;

    while (true) {
        c += 1;
        y += 1;
        z += 2;
        const yc = y * c;
        const num_k = num_km1 * z - num_km2 * yc;
        const den_k = den_km1 * z - den_km2 * yc;

        if (den_k != 0) {
            const r = num_k / den_k;
            if (r == frac) {
                return factor * frac;
            }
            frac = r;
        }

        num_km2 = num_km1;
        den_km2 = den_km1;
        num_km1 = num_k;
        den_km1 = den_k;
        if (@abs(num_k) > big) {
            num_km2 *= small;
            den_km2 *= small;
            num_km1 *= small;
            den_km1 *= small;
        }
    }
}

pub fn quantile(p: f64, a: f64) f64 {
    if (p <= 0) {
        return 0;
    }
    if (p >= 1) {
        return std.math.inf(f64);
    }

    const lgamma = std.math.lgamma(f64, a);
    const dithresh = 5 * small;
    const y0 = 1 - p;

    var x_high = max_float;
    var x_low: f64 = 0;
    var y_high: f64 = 1;
    var y_low: f64 = 0;

    var x = blk: {
        const d = 1 / (9 * a);
        const t = 1 - d - normal.quantile(y0) * @sqrt(d);
        break :blk a * t * t * t;
    };
    var y = survival(x, a);

    for (0..10) |_| {
        if (x < x_low or x > x_high) {
            break;
        }
        if (y < y_low or y > y_high) {
            break;
        }

        if (y < y0) {
            x_high = x;
            y_low = y;
        } else {
            x_low = x;
            y_high = y;
        }

        const d = blk: {
            const log_density = (a - 1) * @log(x) - x - lgamma;
            if (log_density < exp_underflow) {
                break;
            }
            const density = @exp(log_density);
            const d = (y0 - y) / density;
            if (@abs(d / x) < small) {
                return x;
            }
            break :blk d;
        };

        x -= d;
        y = survival(x, a);
    }

    if (x_high == max_float) {
        if (x < 0) {
            x = 1;
        }
        var d: f64 = 0.0625;
        while (x_high == max_float) : (d *= 2) {
            x = (1 + d) * x;
            y = survival(x, a);
            if (y < y0) {
                x_high = x;
                y_low = y;
                break;
            }
        }
    }

    var d: f64 = 0.5;
    var dir: f64 = 0;
    for (0..400) |_| {
        x = x_low + d * (x_high - x_low);
        y = survival(x, a);

        if (@abs((x_high - x_low) / (x_high + x_low)) < dithresh) {
            break;
        }
        if (@abs((y - y0) / y0) < dithresh) {
            break;
        }
        if (x <= 0) {
            break;
        }

        if (y >= y0) {
            x_low = x;
            y_high = y;

            if (dir < 0) {
                dir = 0;
                d = 0.5;
            } else if (dir > 1) {
                d = 0.5 * d + 0.5;
            } else {
                d = (y0 - y_low) / (y_high - y_low);
            }

            dir += 1;
        } else {
            x_high = x;
            y_low = y;

            if (dir > 0) {
                dir = 0;
                d = 0.5;
            } else if (dir < -1) {
                d = 0.5 * d;
            } else {
                d = (y0 - y_low) / (y_high - y_low);
            }

            dir -= 1;
        }
    }
    return x;
}
