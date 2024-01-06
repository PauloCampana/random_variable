const descriptive = @import("../descriptive.zig");

/// Sum of squares: total
pub fn sst(dependent: []const f64) f64 {
    const mean = descriptive.mean.arithmetic(dependent);
    var sum: f64 = 0;
    for (dependent) |y| {
        const d = y - mean;
        sum += d * d;
    }
    return sum;
}

/// Mean of squares: total
pub fn mst(dependent: []const f64) f64 {
    const df = @as(f64, @floatFromInt(dependent.len)) - 1;
    return sst(dependent) / df;
}

/// Sum of squares: error
pub fn sse(residue: []const f64) f64 {
    var sum: f64 = 0;
    for (residue) |e| {
        sum += e * e;
    }
    return sum;
}

/// Mean of squares: error
pub fn mse(residue: []const f64, p: usize) f64 {
    const df = @as(f64, @floatFromInt(residue.len - p));
    return sse(residue) / df;
}

/// Sum of squares: regression
pub fn ssr(dependent: []const f64, prediction: []const f64) f64 {
    const mean = descriptive.mean.arithmetic(dependent);
    var sum: f64 = 0;
    for (prediction) |p| {
        const d = p - mean;
        sum += d * d;
    }
    return sum;
}

/// Mean of squares: regression
pub fn msr(dependent: []const f64, prediction: []const f64, p: usize) f64 {
    const df = @as(f64, @floatFromInt(p - 1));
    return ssr(dependent, prediction) / df;
}
