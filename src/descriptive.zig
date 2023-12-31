//! Descriptive statistics

const std = @import("std");
const assert = std.debug.assert;

/// Convert a slice of bool/int/float into a slice of `f64`.
pub fn toFloat(allocator: std.mem.Allocator, slice: anytype) ![]f64 {
    const result = try allocator.alloc(f64, slice.len);
    for (result, slice) |*x, y| {
        x.* = switch (@typeInfo(@TypeOf(y))) {
            .Bool => @floatFromInt(@intFromBool(y)),
            .Int => @floatFromInt(y),
            .Float => @floatCast(y),
            else => @compileError("bad type"),
        };
    }
    return result;
}

/// Sample size as a `f64` instead of `usize`.
pub fn length(slice: []const f64) f64 {
    return @floatFromInt(slice.len);
}

/// Sample sum of the elements.
pub fn sum(slice: []const f64) f64 {
    var sum1: f64 = 0;
    for (slice) |x| {
        sum1 += x;
    }
    return sum1;
}

/// Sample product of the elements.
pub fn product(slice: []const f64) f64 {
    var prod: f64 = 1;
    for (slice) |x| {
        prod *= x;
    }
    return prod;
}

/// Sample mean
pub const mean = struct {
    /// Arithmetic mean.
    pub fn arithmetic(slice: []const f64) f64 {
        const n = length(slice);
        const sum1 = sum(slice);
        return sum1 / n;
    }

    /// Geometric mean,
    /// elements must not be negative.
    pub fn geometric(slice: []const f64) f64 {
        const n = length(slice);
        var sum_log: f64 = 0;
        for (slice) |x| {
            sum_log += @log(x);
        }
        return @exp(sum_log / n);
    }

    /// Harmonic mean.
    pub fn harmonic(slice: []const f64) f64 {
        const n = length(slice);
        var sum_inv: f64 = 0;
        for (slice) |x| {
            sum_inv += 1 / x;
        }
        return n / sum_inv;
    }
};

pub fn quantileInPlace(slice: []f64, p: f64) f64 {
    std.mem.sortUnstable(f64, slice, {}, std.sort.asc(f64));
    if (p == 1) {
        return slice[slice.len - 1];
    }
    const index = p * length(slice);
    return slice[@intFromFloat(index)];
}

pub fn quantileAlloc(allocator: std.mem.Allocator, slice: []const f64, p: f64) !f64 {
    const copy = try allocator.dupe(f64, slice);
    defer allocator.free(copy);
    return quantileInPlace(copy, p);
}

pub fn medianInPlace(slice: []f64) f64 {
    return quantileInPlace(slice, 0.5);
}

pub fn medianAlloc(allocator: std.mem.Allocator, slice: []const f64) !f64 {
    return quantileAlloc(allocator, slice, 0.5);
}

/// Sample variance
pub fn variance(slice: []const f64) f64 {
    const n = length(slice);
    const sum1 = sum(slice);
    var sum2: f64 = 0;
    for (slice) |x| {
        sum2 += x * x;
    }
    return (sum2 - sum1 * sum1 / n) / (n - 1);
}

/// Sample standard deviation
pub fn standardDeviation(slice: []const f64) f64 {
    const s2 = variance(slice);
    return @sqrt(s2);
}

/// Sample standard error
pub fn standardError(slice: []const f64) f64 {
    const n = length(slice);
    const s2 = variance(slice);
    return @sqrt(s2 / n);
}

/// Sample skewness
pub fn skewness(slice: []const f64) f64 {
    const n = length(slice);
    const m = mean.arithmetic(slice);
    var sumd2: f64 = 0;
    var sumd3: f64 = 0;
    for (slice) |x| {
        const d = x - m;
        sumd2 += d * d;
        sumd3 += d * d * d;
    }
    const num = sumd3 / n;
    const den = @sqrt(sumd2 / n);
    return num / (den * den * den);
}

/// Sample kurtosis
pub fn kurtosis(slice: []const f64) f64 {
    const n = length(slice);
    const m = mean.arithmetic(slice);
    var sumd2: f64 = 0;
    var sumd4: f64 = 0;
    for (slice) |x| {
        const d = x - m;
        const d2 = d * d;
        sumd2 += d2;
        sumd4 += d2 * d2;
    }
    return n * sumd4 / (sumd2 * sumd2);
}

/// Sample covariance
pub fn covariance(slice1: []const f64, slice2: []const f64) f64 {
    assert(slice1.len == slice2.len);
    const n = length(slice1);
    var sumx: f64 = 0;
    var sumy: f64 = 0;
    var sumxy: f64 = 0;
    for (slice1, slice2) |x, y| {
        sumx += x;
        sumy += y;
        sumxy += x * y;
    }
    return (sumxy - sumx * sumy / n) / (n - 1);
}

/// Sample correlation,
/// ranges from -1 (perfect inverse correlation)
/// to ±0 (no correlation)
/// to +1 (perfect correlation).
pub const correlation = struct {
    /// Standard correlation for linear relationships.
    pub fn pearson(slice1: []const f64, slice2: []const f64) f64 {
        assert(slice1.len == slice2.len);
        const n = length(slice1);
        var sumx: f64 = 0;
        var sumy: f64 = 0;
        var sumxy: f64 = 0;
        var sumxx: f64 = 0;
        var sumyy: f64 = 0;
        for (slice1, slice2) |x, y| {
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

    /// Ranks starts at 1, averages them in case of ties.
    fn rank(allocator: std.mem.Allocator, slice: []const f64) ![]f64 {
        const sorted = try allocator.dupe(f64, slice);
        defer allocator.free(sorted);
        std.mem.sortUnstable(f64, sorted, {}, std.sort.asc(f64));
        const ranked = try allocator.dupe(f64, slice);
        errdefer allocator.free(ranked);
        for (ranked) |*r| {
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
        return ranked;
    }

    /// Concordance based correlation for monotonic relationships
    pub fn kendall(slice1: []const f64, slice2: []const f64) f64 {
        assert(slice1.len == slice2.len);
        var net_concordant: f64 = 0;
        var ties1: f64 = 0;
        var ties2: f64 = 0;
        for (slice1, slice2, 0..) |xi, yi, i| {
            for (slice1[i + 1..], slice2[i + 1..]) |xj, yj| {
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
        const n = length(slice1);
        const combinations = n * (n - 1) / 2;
        const den1 = combinations - ties1;
        const den2 = combinations - ties2;
        return net_concordant / @sqrt(den1 * den2);
    }

    /// Rank based correlation for monotonic relationships.
    pub fn spearman(allocator: std.mem.Allocator, slice1: []const f64, slice2: []const f64) !f64 {
        assert(slice1.len == slice2.len);
        const rank1 = try rank(allocator, slice1);
        const rank2 = try rank(allocator, slice2);
        defer allocator.free(rank1);
        defer allocator.free(rank2);
        return pearson(rank1, rank2);
    }
};

const ta = std.testing.allocator;
const expectEqual = std.testing.expectEqual;
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
    try expectEqual(@as(f64, 11), length(anscombe.x1));
}

test "descriptive.sum" {
    try expectApproxEqRel(@as(f64, 99   ), sum(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64, 99   ), sum(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64, 99   ), sum(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64, 99   ), sum(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64, 82.51), sum(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 82.51), sum(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 82.5 ), sum(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 82.51), sum(anscombe.y4), eps);
}

test "descriptive.product" {
    try expectApproxEqRel(@as(f64, 14529715200       ), product(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64, 14529715200       ), product(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64, 14529715200       ), product(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64, 20401094656       ), product(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64,  2833760410.777853), product(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64,  2553792107.502125), product(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64,  3103197573.579096), product(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64,  3063027549.622185), product(anscombe.y4), eps);
}

test "descriptive.mean.arithmetic" {
    try expectApproxEqRel(@as(f64, 9                ), mean.arithmetic(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64, 9                ), mean.arithmetic(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64, 9                ), mean.arithmetic(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64, 9                ), mean.arithmetic(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64, 7.500909090909091), mean.arithmetic(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 7.500909090909091), mean.arithmetic(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 7.5              ), mean.arithmetic(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 7.500909090909091), mean.arithmetic(anscombe.y4), eps);
}

test "descriptive.mean.geometric" {
    try expectApproxEqRel(@as(f64, 8.391537790124792), mean.geometric(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64, 8.391537790124792), mean.geometric(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64, 8.391537790124792), mean.geometric(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64, 8.654484902031250), mean.geometric(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64, 7.232788054567219), mean.geometric(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 7.164711138020261), mean.geometric(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 7.292757391223743), mean.geometric(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 7.284124410831672), mean.geometric(anscombe.y4), eps);
}

test "descriptive.mean.harmonic" {
    try expectApproxEqRel(@as(f64, 7.756152252222285), mean.harmonic(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64, 7.756152252222285), mean.harmonic(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64, 7.756152252222285), mean.harmonic(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64, 8.444444444444445), mean.harmonic(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64, 6.949772613414747), mean.harmonic(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 6.723295789626634), mean.harmonic(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 7.119524800614420), mean.harmonic(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 7.094886958518605), mean.harmonic(anscombe.y4), eps);
}

test "descriptive.variance" {
    try expectApproxEqRel(@as(f64, 11               ), variance(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64, 11               ), variance(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64, 11               ), variance(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64, 11               ), variance(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64, 4.127269090909091), variance(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 4.127629090909091), variance(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 4.12262          ), variance(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 4.123249090909091), variance(anscombe.y4), eps);
}

test "descriptive.standardDeviation" {
    try expectApproxEqRel(@as(f64, 3.316624790355400), standardDeviation(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64, 3.316624790355400), standardDeviation(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64, 3.316624790355400), standardDeviation(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64, 3.316624790355400), standardDeviation(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64, 2.031568135925815), standardDeviation(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 2.031656735501618), standardDeviation(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 2.030423601123667), standardDeviation(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 2.030578511387602), standardDeviation(anscombe.y4), eps);
}

test "descriptive.standardError" {
    try expectApproxEqRel(@as(f64, 1                 ), standardError(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64, 1                 ), standardError(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64, 1                 ), standardError(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64, 1                 ), standardError(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64, 0.6125408402643334), standardError(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 0.6125675540415626), standardError(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 0.6121957500372217), standardError(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 0.6122424572391897), standardError(anscombe.y4), eps);
}

test "descriptive.skewness" {
    try expectApproxEqRel(@as(f64,  0                  ), skewness(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64,  0                  ), skewness(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64,  0                  ), skewness(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64,  2.84604989415154153), skewness(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64, -0.05580806588592109), skewness(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, -1.12910800171669234), skewness(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64,  1.59223073581644159), skewness(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64,  1.29302528963786001), skewness(anscombe.y4), eps);
}

test "descriptive.kurtosis" {
    try expectApproxEqRel(@as(f64, 1.78             ), kurtosis(anscombe.x1), eps);
    try expectApproxEqRel(@as(f64, 1.78             ), kurtosis(anscombe.x2), eps);
    try expectApproxEqRel(@as(f64, 1.78             ), kurtosis(anscombe.x3), eps);
    try expectApproxEqRel(@as(f64, 9.1              ), kurtosis(anscombe.x4), eps);
    try expectApproxEqRel(@as(f64, 2.179061359376356), kurtosis(anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 3.007673939693123), kurtosis(anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 5.130453167839065), kurtosis(anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 4.390788953777712), kurtosis(anscombe.y4), eps);
}

test "descriptive.covariance" {
    try expectApproxEqRel(@as(f64, 5.501), covariance(anscombe.x1, anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 5.5  ), covariance(anscombe.x2, anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 5.497), covariance(anscombe.x3, anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 5.499), covariance(anscombe.x4, anscombe.y4), eps);
}

test "descriptive.correlation.pearson" {
    try expectApproxEqRel(@as(f64, 0.8164205163448399), correlation.pearson(anscombe.x1, anscombe.y1), eps);
    try expectApproxEqRel(@as(f64, 0.8162365060002429), correlation.pearson(anscombe.x2, anscombe.y2), eps);
    try expectApproxEqRel(@as(f64, 0.8162867394895982), correlation.pearson(anscombe.x3, anscombe.y3), eps);
    try expectApproxEqRel(@as(f64, 0.8165214368885029), correlation.pearson(anscombe.x4, anscombe.y4), eps);
}
