const std = @import("std");

pub fn sum(comptime T: type, slice: []T) T {
    var s: T = 0;
    for (slice) |x| {
        s += x;
    }
    return s;
}

pub fn mean(comptime T: type, slice: []T) f64 {
    const s = std.math.lossyCast(f64, sum(T, slice));
    const len = @as(f64, @floatFromInt(slice.len));
    return s / len;
}

pub fn median(comptime T: type, slice: []T) !T {
    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const temp = try allocator.dupe(T, slice);
    defer allocator.free(temp);
    std.mem.sortUnstable(T, temp, {}, std.sort.asc(T));
    return temp[temp.len / 2];
}

pub fn medianInPlace(comptime T: type, slice: []T) T {
    std.mem.sortUnstable(T, slice, {}, std.sort.asc(T));
    return slice[slice.len / 2];
}

pub fn variance(comptime T: type, slice: []T) f64 {
    const len = @as(f64, @floatFromInt(slice.len));
    const m = mean(T, slice);
    var s: f64 = 0;
    for (slice) |x| {
        const xx = std.math.lossyCast(f64, x * x);
        s += xx;
    }
    return (s - len * m * m) / (len - 1);
}

pub fn standardDeviation(comptime T: type, slice: []T) f64 {
    return @sqrt(variance(T, slice));
}

pub fn standardError(comptime T: type, slice: []T) f64 {
    const len = @as(f64, @floatFromInt(slice.len));
    return @sqrt(variance(T, slice) / len);
}

pub fn skewness(comptime T: type, slice: []T) f64 {
    const len = @as(f64, @floatFromInt(slice.len));
    const m = mean(T, slice);
    const sd = standardDeviation(T, slice);
    var s: f64 = 0;
    for (slice) |x| {
        const fx = std.math.lossyCast(f64, x);
        const d = fx - m;
        s += d * d * d;
    }
    return s / (len * sd * sd * sd);
}

pub fn kurtosis(comptime T: type, slice: []T) f64 {
    const len = @as(f64, @floatFromInt(slice.len));
    const m = mean(T, slice);
    var sum2: f64 = 0;
    var sum4: f64 = 0;
    for (slice) |x| {
        const fx = std.math.lossyCast(f64, x);
        const d = fx - m;
        const d2 = d * d;
        sum2 += d2;
        sum4 += d2 * d2;
    }
    return len * sum4 / (sum2 * sum2);
}

pub fn covariance(comptime T: type, slice1: []T, slice2: []T) f64 {
    const len = @as(f64, @floatFromInt(slice1.len));
    const m1 = mean(T, slice1);
    const m2 = mean(T, slice2);
    var s: f64 = 0;
    for (slice1, slice2) |x, y| {
        const fx = std.math.lossyCast(f64, x);
        const fy = std.math.lossyCast(f64, y);
        s += fx * fy;
    }
    return (s - len * m1 * m2) / (len - 1);
}

pub fn correlation(comptime T: type, slice1: []T, slice2: []T) f64 {
    const len = @as(f64, @floatFromInt(slice1.len));
    const m1 = mean(T, slice1);
    const m2 = mean(T, slice2);
    var sumxy: f64 = 0;
    var sumxx: f64 = 0;
    var sumyy: f64 = 0;
    for (slice1, slice2) |x, y| {
        const fx = std.math.lossyCast(f64, x);
        const fy = std.math.lossyCast(f64, y);
        sumxy += fx * fy;
        sumxx += fx * fx;
        sumyy += fy * fy;
    }
    const num  = sumxy - len * m1 * m2;
    const den1 = sumxx - len * m1 * m1;
    const den2 = sumyy - len * m2 * m2;
    return num / @sqrt(den1 * den2);
}
