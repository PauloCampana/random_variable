//! Statistical functions

pub const distribution = @import("distribution.zig");

pub const descriptive = @import("descriptive.zig");
pub const hypotesis = @import("hypothesis.zig");

pub const Matrix = @import("Matrix.zig");
pub const csv = @import("csv.zig");
pub const LinearModel = @import("LinearModel.zig");

test {
    @import("std").testing.refAllDeclsRecursive(@This());
}
