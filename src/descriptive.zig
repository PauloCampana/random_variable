//! Estimators for common descriptive statistics.

const std = @import("std");

/// Convert a slice of bool/int/float into a slice of `f64`.
pub fn toFloat(allocator: std.mem.Allocator, slice: anytype) ![]f64 {
    const result = try allocator.alloc(f64, slice.len);
    for (result, slice) |*x, y| {
        x.* = switch (@typeInfo(@TypeOf(y))) {
            .Bool => @floatFromInt(@intFromBool(y)),
            .Int => @floatFromInt(y),
            .Float => @floatCast(y),
            else => @compileError("bad slice type"),
        };
    }
    return result;
}

/// Sample size as a `f64` instead of `usize`.
pub fn length(sample: []const f64) f64 {
    return @floatFromInt(sample.len);
}

/// Sample sum.
pub fn sum(sample: []const f64) f64 {
    var sum1: f64 = 0;
    for (sample) |x| {
        sum1 += x;
    }
    return sum1;
}

/// Sample product.
pub fn product(sample: []const f64) f64 {
    var prod: f64 = 1;
    for (sample) |x| {
        prod *= x;
    }
    return prod;
}

/// Estimators for sample position measures.
pub const mean = struct {
    /// Arithmetic mean.
    pub fn arithmetic(sample: []const f64) f64 {
        const n = length(sample);
        const sum1 = sum(sample);
        return sum1 / n;
    }

    /// Geometric mean,
    /// elements must not be negative.
    pub fn geometric(sample: []const f64) f64 {
        const n = length(sample);
        var sum_log: f64 = 0;
        for (sample) |x| {
            sum_log += @log(x);
        }
        return @exp(sum_log / n);
    }

    /// Harmonic mean.
    pub fn harmonic(sample: []const f64) f64 {
        const n = length(sample);
        var sum_inv: f64 = 0;
        for (sample) |x| {
            sum_inv += 1 / x;
        }
        return n / sum_inv;
    }
};

/// Estimator for sample quantiles.
pub fn quantile(allocator: std.mem.Allocator, sample: []const f64, p: f64) !f64 {
    std.debug.assert(0 <= p and p <= 1);
    const sorted = try allocator.dupe(f64, sample);
    defer allocator.free(sorted);
    std.mem.sortUnstable(f64, sorted, {}, std.sort.asc(f64));
    const virtual_index = p * (length(sorted) - 1);
    const lower = sorted[@intFromFloat(@floor(virtual_index))];
    const upper = sorted[@intFromFloat(@ceil(virtual_index))];
    const fractional_part = @rem(virtual_index, 1);
    return lower + (upper - lower) * fractional_part;
}

/// Estimator for sample median.
pub fn median(allocator: std.mem.Allocator, sample: []const f64) !f64 {
    return quantile(allocator, sample, 0.5);
}

/// Estimator for sample minimum value.
pub fn min(sample: []const f64) f64 {
    var result = sample[0];
    for (sample[1..]) |x| {
        if (x < result) {
            result = x;
        }
    }
    return result;
}

/// Estimator for sample maximum value.
pub fn max(sample: []const f64) f64 {
    var result = sample[0];
    for(sample[1..]) |x| {
        if (x > result) {
            result = x;
        }
    }
    return result;
}

/// Estimator for sample variance.
pub fn variance(sample: []const f64) f64 {
    const n = length(sample);
    var sum1: f64 = 0;
    var sum2: f64 = 0;
    for (sample) |x| {
        sum1 += x;
        sum2 += x * x;
    }
    return (sum2 - sum1 * sum1 / n) / (n - 1);
}

/// Estimator for sample standard deviation.
pub fn standardDeviation(sample: []const f64) f64 {
    const s2 = variance(sample);
    return @sqrt(s2);
}

/// Estimator for sample standard deviation or the mean.
pub fn standardError(sample: []const f64) f64 {
    const n = length(sample);
    const s2 = variance(sample);
    return @sqrt(s2 / n);
}

/// Estimator for sample third standardized moment (biased).
pub fn skewness(sample: []const f64) f64 {
    const n = length(sample);
    const m = mean.arithmetic(sample);
    var sumd2: f64 = 0;
    var sumd3: f64 = 0;
    for (sample) |x| {
        const d = x - m;
        sumd2 += d * d;
        sumd3 += d * d * d;
    }
    const num = sumd3 / n;
    const den = @sqrt(sumd2 / n);
    return num / (den * den * den);
}

/// Estimator for sample fourth standardized moment (biased).
pub fn kurtosis(sample: []const f64) f64 {
    const n = length(sample);
    const m = mean.arithmetic(sample);
    var sumd2: f64 = 0;
    var sumd4: f64 = 0;
    for (sample) |x| {
        const d = x - m;
        const d2 = d * d;
        sumd2 += d2;
        sumd4 += d2 * d2;
    }
    return n * sumd4 / (sumd2 * sumd2);
}

/// Estimator for sample covariance (pearson).
pub fn covariance(sample1: []const f64, sample2: []const f64) f64 {
    const n = length(sample1);
    var sumx: f64 = 0;
    var sumy: f64 = 0;
    var sumxy: f64 = 0;
    for (sample1, sample2) |x, y| {
        sumx += x;
        sumy += y;
        sumxy += x * y;
    }
    return (sumxy - sumx * sumy / n) / (n - 1);
}

/// Estimators for sample correlation.
pub const correlation = struct {
    /// Standard correlation for linear relationships.
    pub fn pearson(sample1: []const f64, sample2: []const f64) f64 {
        std.debug.assert(sample1.len == sample2.len);
        const n = length(sample1);
        var sumx: f64 = 0;
        var sumy: f64 = 0;
        var sumxy: f64 = 0;
        var sumxx: f64 = 0;
        var sumyy: f64 = 0;
        for (sample1, sample2) |x, y| {
            sumx += x;
            sumy += y;
            sumxx += x * x;
            sumyy += y * y;
            sumxy += x * y;
        }
        const num  = sumxy - sumx * sumy / n;
        const den1 = sumxx - sumx * sumx / n;
        const den2 = sumyy - sumy * sumy / n;
        return num / @sqrt(den1 * den2);
    }

    /// Concordance based correlation for monotonic relationships.
    pub fn kendall(sample1: []const f64, sample2: []const f64) f64 {
        std.debug.assert(sample1.len == sample2.len);
        var net_concordant: f64 = 0;
        var ties1: f64 = 0;
        var ties2: f64 = 0;
        for (sample1, sample2, 0..) |xi, yi, i| {
            for (sample1[i + 1..], sample2[i + 1..]) |xj, yj| {
                if (xi == xj) {
                    ties1 += 1;
                    if (yi == yj) {
                        ties2 += 2;
                    }
                    continue;
                }
                if ((xi < xj and yi < yj) or (xi > xj and yi > yj)) {
                    net_concordant += 1;
                } else {
                    net_concordant -= 1;
                }
            }
        }
        const n = length(sample1);
        const combinations = n * (n - 1) / 2;
        const den1 = combinations - ties1;
        const den2 = combinations - ties2;
        return net_concordant / @sqrt(den1 * den2);
    }

    /// Rank based correlation for monotonic relationships.
    pub fn spearman(allocator: std.mem.Allocator, sample1: []const f64, sample2: []const f64) !f64 {
        std.debug.assert(sample1.len == sample2.len);
        const rank1 = try rank(allocator, sample1);
        const rank2 = try rank(allocator, sample2);
        defer allocator.free(rank1);
        defer allocator.free(rank2);
        return pearson(rank1, rank2);
    }

    /// Ranks starts at 1, averages them in case of ties.
    fn rank(allocator: std.mem.Allocator, sample: []const f64) ![]f64 {
        const sorted = try allocator.dupe(f64, sample);
        defer allocator.free(sorted);
        std.mem.sortUnstable(f64, sorted, {}, std.sort.asc(f64));
        const result = try allocator.dupe(f64, sample);
        for (result) |*r| {
            var rank_sum: f64 = 0;
            var times_summed: f64 = 0;
            for (sorted, 1..) |s, i| {
                if (r.* == s) {
                    rank_sum += @floatFromInt(i);
                    times_summed += 1;
                }
            }
            r.* = rank_sum / times_summed;
        }
        return result;
    }
};

const ta = std.testing.allocator;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const expectEqualSlices = std.testing.expectEqualSlices;
const eps = 5e-14;

const anscombe = struct {
    const x1 = &[_]f64 {10   , 8   , 13   , 9   , 11   , 14   , 6   ,  4   , 12   , 7   , 5   };
    const x2 = &[_]f64 {10   , 8   , 13   , 9   , 11   , 14   , 6   ,  4   , 12   , 7   , 5   };
    const x3 = &[_]f64 {10   , 8   , 13   , 9   , 11   , 14   , 6   ,  4   , 12   , 7   , 5   };
    const x4 = &[_]f64 { 8   , 8   ,  8   , 8   ,  8   ,  8   , 8   , 19   ,  8   , 8   , 8   };
    const y1 = &[_]f64 { 8.04, 6.95,  7.58, 8.81,  8.33,  9.96, 7.24,  4.26, 10.84, 4.82, 5.68};
    const y2 = &[_]f64 { 9.14, 8.14,  8.74, 8.77,  9.26,  8.10, 6.13,  3.10,  9.13, 7.26, 4.74};
    const y3 = &[_]f64 { 7.46, 6.77, 12.74, 7.11,  7.81,  8.84, 6.08,  5.39,  8.15, 6.42, 5.73};
    const y4 = &[_]f64 { 6.58, 5.76,  7.71, 8.84,  8.47,  7.04, 5.25,  12.5,  5.56, 7.91, 6.89};
};

test "descriptive.toFloat" {
    const from_bool = try toFloat(ta, &[_]bool {false, false, true});
    const from_u64 = try toFloat(ta, &[_]u64 {3, 5, 9});
    const from_i64 = try toFloat(ta, &[_]i64 {3, 5, 9});
    const from_f32 = try toFloat(ta, &[_]f32 {3, 5, 9});
    defer ta.free(from_bool);
    defer ta.free(from_u64);
    defer ta.free(from_i64);
    defer ta.free(from_f32);
    try expectEqualSlices(f64, &.{0, 0, 1}, from_bool);
    try expectEqualSlices(f64, &.{3, 5, 9}, from_u64);
    try expectEqualSlices(f64, &.{3, 5, 9}, from_i64);
    try expectEqualSlices(f64, &.{3, 5, 9}, from_f32);
}

test "descriptive.lenght" {
    try expectApproxEqRel(11, length(anscombe.x1), eps);
}

test "descriptive.sum" {
    try expectApproxEqRel(99   , sum(anscombe.x1), eps);
    try expectApproxEqRel(99   , sum(anscombe.x2), eps);
    try expectApproxEqRel(99   , sum(anscombe.x3), eps);
    try expectApproxEqRel(99   , sum(anscombe.x4), eps);
    try expectApproxEqRel(82.51, sum(anscombe.y1), eps);
    try expectApproxEqRel(82.51, sum(anscombe.y2), eps);
    try expectApproxEqRel(82.5 , sum(anscombe.y3), eps);
    try expectApproxEqRel(82.51, sum(anscombe.y4), eps);
}

test "descriptive.product" {
    try expectApproxEqRel(14529715200       , product(anscombe.x1), eps);
    try expectApproxEqRel(14529715200       , product(anscombe.x2), eps);
    try expectApproxEqRel(14529715200       , product(anscombe.x3), eps);
    try expectApproxEqRel(20401094656       , product(anscombe.x4), eps);
    try expectApproxEqRel( 2833760410.777853, product(anscombe.y1), eps);
    try expectApproxEqRel( 2553792107.502125, product(anscombe.y2), eps);
    try expectApproxEqRel( 3103197573.579096, product(anscombe.y3), eps);
    try expectApproxEqRel( 3063027549.622185, product(anscombe.y4), eps);
}

test "descriptive.mean.arithmetic" {
    try expectApproxEqRel(9                , mean.arithmetic(anscombe.x1), eps);
    try expectApproxEqRel(9                , mean.arithmetic(anscombe.x2), eps);
    try expectApproxEqRel(9                , mean.arithmetic(anscombe.x3), eps);
    try expectApproxEqRel(9                , mean.arithmetic(anscombe.x4), eps);
    try expectApproxEqRel(7.500909090909091, mean.arithmetic(anscombe.y1), eps);
    try expectApproxEqRel(7.500909090909091, mean.arithmetic(anscombe.y2), eps);
    try expectApproxEqRel(7.5              , mean.arithmetic(anscombe.y3), eps);
    try expectApproxEqRel(7.500909090909091, mean.arithmetic(anscombe.y4), eps);
}

test "descriptive.mean.geometric" {
    try expectApproxEqRel(8.391537790124792, mean.geometric(anscombe.x1), eps);
    try expectApproxEqRel(8.391537790124792, mean.geometric(anscombe.x2), eps);
    try expectApproxEqRel(8.391537790124792, mean.geometric(anscombe.x3), eps);
    try expectApproxEqRel(8.654484902031250, mean.geometric(anscombe.x4), eps);
    try expectApproxEqRel(7.232788054567219, mean.geometric(anscombe.y1), eps);
    try expectApproxEqRel(7.164711138020261, mean.geometric(anscombe.y2), eps);
    try expectApproxEqRel(7.292757391223743, mean.geometric(anscombe.y3), eps);
    try expectApproxEqRel(7.284124410831672, mean.geometric(anscombe.y4), eps);
}

test "descriptive.mean.harmonic" {
    try expectApproxEqRel(7.756152252222285, mean.harmonic(anscombe.x1), eps);
    try expectApproxEqRel(7.756152252222285, mean.harmonic(anscombe.x2), eps);
    try expectApproxEqRel(7.756152252222285, mean.harmonic(anscombe.x3), eps);
    try expectApproxEqRel(8.444444444444445, mean.harmonic(anscombe.x4), eps);
    try expectApproxEqRel(6.949772613414747, mean.harmonic(anscombe.y1), eps);
    try expectApproxEqRel(6.723295789626634, mean.harmonic(anscombe.y2), eps);
    try expectApproxEqRel(7.119524800614420, mean.harmonic(anscombe.y3), eps);
    try expectApproxEqRel(7.094886958518605, mean.harmonic(anscombe.y4), eps);
}

test "descriptive.quantile" {
    try expectApproxEqRel( 4.26, try quantile(ta, anscombe.y1, 0   ), eps);
    try expectApproxEqRel( 4.54, try quantile(ta, anscombe.y1, 0.05), eps);
    try expectApproxEqRel( 4.82, try quantile(ta, anscombe.y1, 0.1 ), eps);
    try expectApproxEqRel( 9.96, try quantile(ta, anscombe.y1, 0.9 ), eps);
    try expectApproxEqRel(10.4 , try quantile(ta, anscombe.y1, 0.95), eps);
    try expectApproxEqRel(10.84, try quantile(ta, anscombe.y1, 1   ), eps);
}

test "descriptive.min" {
    try expectApproxEqRel(4   , min(anscombe.x1), eps);
    try expectApproxEqRel(4   , min(anscombe.x2), eps);
    try expectApproxEqRel(4   , min(anscombe.x3), eps);
    try expectApproxEqRel(8   , min(anscombe.x4), eps);
    try expectApproxEqRel(4.26, min(anscombe.y1), eps);
    try expectApproxEqRel(3.1 , min(anscombe.y2), eps);
    try expectApproxEqRel(5.39, min(anscombe.y3), eps);
    try expectApproxEqRel(5.25, min(anscombe.y4), eps);
}

test "descriptive.max" {
    try expectApproxEqRel(14   , max(anscombe.x1), eps);
    try expectApproxEqRel(14   , max(anscombe.x2), eps);
    try expectApproxEqRel(14   , max(anscombe.x3), eps);
    try expectApproxEqRel(19   , max(anscombe.x4), eps);
    try expectApproxEqRel(10.84, max(anscombe.y1), eps);
    try expectApproxEqRel( 9.26, max(anscombe.y2), eps);
    try expectApproxEqRel(12.74, max(anscombe.y3), eps);
    try expectApproxEqRel(12.5 , max(anscombe.y4), eps);
}

test "descriptive.variance" {
    try expectApproxEqRel(11               , variance(anscombe.x1), eps);
    try expectApproxEqRel(11               , variance(anscombe.x2), eps);
    try expectApproxEqRel(11               , variance(anscombe.x3), eps);
    try expectApproxEqRel(11               , variance(anscombe.x4), eps);
    try expectApproxEqRel(4.127269090909091, variance(anscombe.y1), eps);
    try expectApproxEqRel(4.127629090909091, variance(anscombe.y2), eps);
    try expectApproxEqRel(4.12262          , variance(anscombe.y3), eps);
    try expectApproxEqRel(4.123249090909091, variance(anscombe.y4), eps);
}

test "descriptive.standardDeviation" {
    try expectApproxEqRel(3.316624790355400, standardDeviation(anscombe.x1), eps);
    try expectApproxEqRel(3.316624790355400, standardDeviation(anscombe.x2), eps);
    try expectApproxEqRel(3.316624790355400, standardDeviation(anscombe.x3), eps);
    try expectApproxEqRel(3.316624790355400, standardDeviation(anscombe.x4), eps);
    try expectApproxEqRel(2.031568135925815, standardDeviation(anscombe.y1), eps);
    try expectApproxEqRel(2.031656735501618, standardDeviation(anscombe.y2), eps);
    try expectApproxEqRel(2.030423601123667, standardDeviation(anscombe.y3), eps);
    try expectApproxEqRel(2.030578511387602, standardDeviation(anscombe.y4), eps);
}

test "descriptive.standardError" {
    try expectApproxEqRel(1                 , standardError(anscombe.x1), eps);
    try expectApproxEqRel(1                 , standardError(anscombe.x2), eps);
    try expectApproxEqRel(1                 , standardError(anscombe.x3), eps);
    try expectApproxEqRel(1                 , standardError(anscombe.x4), eps);
    try expectApproxEqRel(0.6125408402643334, standardError(anscombe.y1), eps);
    try expectApproxEqRel(0.6125675540415626, standardError(anscombe.y2), eps);
    try expectApproxEqRel(0.6121957500372217, standardError(anscombe.y3), eps);
    try expectApproxEqRel(0.6122424572391897, standardError(anscombe.y4), eps);
}

test "descriptive.skewness" {
    try expectApproxEqRel( 0                  , skewness(anscombe.x1), eps);
    try expectApproxEqRel( 0                  , skewness(anscombe.x2), eps);
    try expectApproxEqRel( 0                  , skewness(anscombe.x3), eps);
    try expectApproxEqRel( 2.84604989415154153, skewness(anscombe.x4), eps);
    try expectApproxEqRel(-0.05580806588592109, skewness(anscombe.y1), eps);
    try expectApproxEqRel(-1.12910800171669234, skewness(anscombe.y2), eps);
    try expectApproxEqRel( 1.59223073581644159, skewness(anscombe.y3), eps);
    try expectApproxEqRel( 1.29302528963786001, skewness(anscombe.y4), eps);
}

test "descriptive.kurtosis" {
    try expectApproxEqRel(1.78             , kurtosis(anscombe.x1), eps);
    try expectApproxEqRel(1.78             , kurtosis(anscombe.x2), eps);
    try expectApproxEqRel(1.78             , kurtosis(anscombe.x3), eps);
    try expectApproxEqRel(9.1              , kurtosis(anscombe.x4), eps);
    try expectApproxEqRel(2.179061359376356, kurtosis(anscombe.y1), eps);
    try expectApproxEqRel(3.007673939693123, kurtosis(anscombe.y2), eps);
    try expectApproxEqRel(5.130453167839065, kurtosis(anscombe.y3), eps);
    try expectApproxEqRel(4.390788953777712, kurtosis(anscombe.y4), eps);
}

test "descriptive.covariance" {
    try expectApproxEqRel(5.501, covariance(anscombe.x1, anscombe.y1), eps);
    try expectApproxEqRel(5.5  , covariance(anscombe.x2, anscombe.y2), eps);
    try expectApproxEqRel(5.497, covariance(anscombe.x3, anscombe.y3), eps);
    try expectApproxEqRel(5.499, covariance(anscombe.x4, anscombe.y4), eps);
}

test "descriptive.correlation.pearson" {
    try expectApproxEqRel(0.8164205163448399, correlation.pearson(anscombe.x1, anscombe.y1), eps);
    try expectApproxEqRel(0.8162365060002429, correlation.pearson(anscombe.x2, anscombe.y2), eps);
    try expectApproxEqRel(0.8162867394895982, correlation.pearson(anscombe.x3, anscombe.y3), eps);
    try expectApproxEqRel(0.8165214368885029, correlation.pearson(anscombe.x4, anscombe.y4), eps);
}

test "descriptive.correlation.kendall" {
    try expectApproxEqRel(0.6363636363636364, correlation.kendall(anscombe.x1, anscombe.y1), eps);
    try expectApproxEqRel(0.5636363636363636, correlation.kendall(anscombe.x2, anscombe.y2), eps);
    try expectApproxEqRel(0.9636363636363636, correlation.kendall(anscombe.x3, anscombe.y3), eps);
    try expectApproxEqRel(0.4264014327112208, correlation.kendall(anscombe.x4, anscombe.y4), eps);
}

test "descriptive.correlation.spearman" {
    try expectApproxEqRel(0.8181818181818182, try correlation.spearman(ta, anscombe.x1, anscombe.y1), eps);
    try expectApproxEqRel(0.6909090909090909, try correlation.spearman(ta, anscombe.x2, anscombe.y2), eps);
    try expectApproxEqRel(0.9909090909090909, try correlation.spearman(ta, anscombe.x3, anscombe.y3), eps);
    try expectApproxEqRel(0.5               , try correlation.spearman(ta, anscombe.x4, anscombe.y4), eps);
}
