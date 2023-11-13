const std = @import("std");
const epsilon = 2.5e-15;

/// flipped arguments so expected coerces to actual
pub fn expectEqual(actual: anytype, expected: @TypeOf(actual)) !void {
    try std.testing.expectEqual(expected, actual);
}

/// flipped arguments so expected coerces to actual
pub fn expectApproxEqRel(actual: anytype, expected: @TypeOf(actual)) !void {
    try std.testing.expectApproxEqRel(expected, actual, epsilon);
}
