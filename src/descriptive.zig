const std = @import("std");

pub fn lossyCastAlloc(comptime T: type, allocator: std.mem.Allocator, slice: anytype) ![]T {
    const out = try allocator.alloc(T, slice.len);
    for (out, slice) |*x, y| {
        x.* = std.math.lossyCast(T, y);
    }
    return out;
}

pub fn length(slice: []f64) f64 {
    return @as(f64, @floatFromInt(slice.len));
}

pub fn sum(slice: []f64) f64 {
    var sumx: f64 = 0;
    for (slice) |x| {
        sumx += x;
    }
    return sumx;
}

pub fn product(slice: []f64) f64 {
    var prodx: f64 = 1;
    for (slice) |x| {
        prodx *= x;
    }
    return prodx;
}

pub fn mean(slice: []f64) f64 {
    const len = length(slice);
    const sumx = sum(slice);
    return sumx / len;
}

pub fn meanGeometric(slice: []f64) f64 {
    const len = length(slice);
    const prodx = product(slice);
    return std.math.pow(f64, prodx, 1 / len);
}

pub fn meanHarmonic(slice: []f64) f64 {
    const len = length(slice);
    var invsum: f64 = 0;
    for (slice) |x| {
        invsum += 1 / x;
    }
    return len / invsum;
}

pub fn medianAlloc(allocator: std.mem.Allocator, slice: []f64) !f64 {
    const temp = try allocator.dupe(f64, slice);
    defer allocator.free(temp);
    std.mem.sortUnstable(f64, temp, {}, std.sort.asc(f64));
    return temp[temp.len / 2];
}

pub fn medianInPlace(slice: []f64) f64 {
    std.mem.sortUnstable(f64, slice, {}, std.sort.asc(f64));
    return slice[slice.len / 2];
}

pub fn variance(slice: []f64) f64 {
    const len = length(slice);
    const sumx = sum(slice);
    var sumxx: f64 = 0;
    for (slice) |x| {
        sumxx += x * x;
    }
    return (sumxx - sumx * sumx / len) / (len - 1);
}

pub fn standardDeviation(slice: []f64) f64 {
    return @sqrt(variance(slice));
}

pub fn standardError(slice: []f64) f64 {
    const len = length(slice);
    return @sqrt(variance(slice) / len);
}

pub fn skewness(slice: []f64) f64 {
    const len = length(slice);
    const meanx = mean(slice);
    const sd = standardDeviation(slice);
    var sumddd: f64 = 0;
    for (slice) |x| {
        const d = x - meanx;
        sumddd += d * d * d;
    }
    return sumddd / (len * sd * sd * sd);
}

pub fn kurtosis(slice: []f64) f64 {
    const len = length(slice);
    const meanx = mean(slice);
    var sumdd: f64 = 0;
    var sumdddd: f64 = 0;
    for (slice) |x| {
        const d = x - meanx;
        const dd = d * d;
        sumdd += dd;
        sumdddd += dd * dd;
    }
    return len * sumdddd / (sumdd * sumdd);
}

pub fn covariance(slice1: []f64, slice2: []f64) f64 {
    const len = length(slice1);
    var sumx: f64 = 0;
    var sumy: f64 = 0;
    var sumxy: f64 = 0;
    for (slice1, slice2) |x, y| {
        sumx += x;
        sumy += y;
        sumxy += x * y;
    }
    return (sumxy - sumx * sumy / len) / (len - 1);
}

pub fn correlation(slice1: []f64, slice2: []f64) f64 {
    const len = length(slice1);
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
    const num  = sumxy - sumx * sumy / len;
    const den1 = sumxx - sumx * sumx / len;
    const den2 = sumyy - sumy * sumy / len;
    return num / @sqrt(den1 * den2);
}

fn rank(allocator: std.mem.Allocator, slice: []f64) ![]f64 {
    const ranked = try allocator.dupe(f64, slice);
    const sorted = try allocator.dupe(f64, slice);
    defer allocator.free(sorted);
    std.mem.sortUnstable(f64, sorted, {}, std.sort.asc(f64));
    const original = slice;
    for (ranked, original) |*r, o| {
        for (sorted, 0..) |s, i| {
            if (o == s) {
                r.* = @as(f64, @floatFromInt(i));
            }
        }
    }
    return ranked;
}

pub fn correlationSpearman(allocator: std.mem.Allocator, slice1: []f64, slice2: []f64) !f64 {
    const rank1 = try rank(allocator, slice1);
    const rank2 = try rank(allocator, slice2);
    defer allocator.free(rank1);
    defer allocator.free(rank2);
    return correlation(rank1, rank2);
}
