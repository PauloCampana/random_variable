//! Functions for manipulation of dynamically allocated matrices

const std = @import("std");
const assert = std.debug.assert;
const Self = @This();

// for testing purposes
var Matrix = Self.setAllocator(std.testing.allocator);
var first = [_]f64 {1, 2, 3};
var secon = [_]f64 {4, 5, 6};
var third = [_]f64 {7, 8, 9};
var all = [_][]f64 {&first, &secon, &third};
const X = Self {.allocator = std.testing.allocator, .data = &all};

allocator: std.mem.Allocator,
data: [][]f64,

/// Specify an allocator for the returning matrices and slices.
pub fn setAllocator(allocator: std.mem.Allocator) Self {
    return Self {
        .allocator = allocator,
        .data = undefined,
    };
}

/// Prints a matrix in the correct orientation,
/// output is customizable with `fmt`, such as "{d:.3}, ".
pub fn print(self: Self, comptime fmt: []const u8) void {
    const printf = std.debug.print;
    const rows = self.data[0].len;
    const cols = self.data.len;
    printf("[{} x {}] {{\n", .{rows, cols});
    for (0..rows) |i| {
        printf("    ", .{});
        for (0..cols) |j| {
            printf(fmt, .{self.data[j][i]});
        }
        printf("\n", .{});
    }
    printf("}}\n", .{});
} // TODO change from debug.print to io.Writer

/// Allocates memory for a rows×columns matrix,
/// the memory is undefined and must first be written to before reading,
/// result must be freed by the caller with `destroy`.
pub fn create(self: Self, rows: usize, cols: usize) !Self {
    const data = try self.allocator.alloc([]f64, cols);
    for (data) |*col| {
        col.* = try self.allocator.alloc(f64, rows);
    }
    return Self {
        .allocator = self.allocator,
        .data = data,
    };
}

/// Frees the allocated memory of a matrix.
pub fn destroy(self: Self) void {
    for (self.data) |col| {
        self.allocator.free(col);
    }
    self.allocator.free(self.data);
}

test "Matrix.create, Matrix.destroy" {
    const Y = try Matrix.create(3, 3);
    defer Y.destroy();
    for (Y.data, X.data) |coly, colx| {
        for (coly, colx) |*y, x| {
            y.* = x;
        }
    }
    try std.testing.expectEqualSlices(f64, &.{1,2,3}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{4,5,6}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{7,8,9}, Y.data[2]);
}

/// Creates a new matrix with the contents of another,
/// result must be freed by the caller with `destroy`.
pub fn dupe(self: Self) !Self {
    const data = try self.allocator.dupe([]f64, self.data);
    for (self.data, data) |old, *new| {
        new.* = try self.allocator.dupe(f64, old);
    }
    return Self {
        .allocator = self.allocator,
        .data = data,
    };
}

test "Matrix.dupe" {
    const Y = try X.dupe();
    defer Y.destroy();
    Y.data[1][1] = 55;
    try std.testing.expectEqual(@as(f64, 55), Y.data[1][1]);
    try std.testing.expectEqual(@as(f64, 5 ), X.data[1][1]);
}

/// Creates a new matrix with the values of a slice,
/// `slice.len` must be divisible by `cols`,
/// result must be freed by the caller with `destroy`.
pub fn createFromSlice(self: Self, slice: []const f64, cols: usize) !Self {
    const rows = try std.math.divExact(usize, slice.len, cols);
    const result = try self.create(rows, cols);
    for (0..cols) |j| {
        const toskip = j * rows;
        for (0..rows) |i| {
            result.data[j][i] = slice[toskip + i];
        }
    }
    return result;
}

test "Matrix.createFromSlice" {
    const Y = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer Y.destroy();
    try std.testing.expectEqualSlices(f64, X.data[0], Y.data[0]);
    try std.testing.expectEqualSlices(f64, X.data[1], Y.data[1]);
    try std.testing.expectEqualSlices(f64, X.data[2], Y.data[2]);
}

/// Creates a new n×n identity matrix,
/// result must be freed by the caller with `destroy`.
pub fn createIdentity(self: Self, n: usize) !Self {
    const result = try self.create(n, n);
    for (0..n) |i| {
        @memset(result.data[i], 0);
        result.data[i][i] = 1;
    }
    return result;
}

test "Matrix.createIdentity" {
    const Y = try Matrix.createIdentity(3);
    defer Y.destroy();
    try std.testing.expectEqualSlices(f64, &.{1, 0, 0}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{0, 1, 0}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{0, 0, 1}, Y.data[2]);
}

/// Creates a new square matrix where the diagonal is taken from a slice,
/// result must be freed by the caller with `destroy`.
pub fn createDiagonal(self: Self, slice: []const f64) !Self {
    const n = slice.len;
    const result = try self.create(n, n);
    for (0..n) |i| {
        @memset(result.data[i], 0);
        result.data[i][i] = slice[i];
    }
    return result;
}

test "Matrix.createDiagonal" {
    const Y = try Matrix.createDiagonal(&.{1, 2, 3});
    defer Y.destroy();
    try std.testing.expectEqualSlices(f64, &.{1, 0, 0}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{0, 2, 0}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{0, 0, 3}, Y.data[2]);
}

/// Returns the diagonal entries of the matrix as a slice,
/// result must be freed by the caller.
pub fn getDiagonal(self: Self) ![]f64 {
    assert(self.data.len == self.data[0].len);
    const n = self.data.len;
    const result = try self.allocator.alloc(f64, n);
    for (0..n) |i| {
        result[i] = self.data[i][i];
    }
    return result;
}

test "Matrix.getDiagonal" {
    const slice = try X.getDiagonal();
    defer std.testing.allocator.free(slice);
    try std.testing.expectEqualSlices(f64, &.{1, 5, 9}, slice);
}

/// Creates a new flipped version of a matrix over its main diagonal,
/// the transpose of a matrix of size a×b is size b×a,
/// result must be freed by the caller with `destroy`.
pub fn transpose(self: Self) !Self {
    const rows = self.data[0].len;
    const cols = self.data.len;
    const result = try self.create(cols, rows);
    for (0..cols) |j| {
        for (0..rows) |i| {
            result.data[j][i] = self.data[i][j];
        }
    }
    return result;
}

test "Matrix.transpose" {
    const Y = try X.transpose();
    defer Y.destroy();
    try std.testing.expectEqualSlices(f64, &.{1, 4, 7}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{2, 5, 8}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{3, 6, 9}, Y.data[2]);
}

/// Calculates the sum of the diagonal entries of a square matrix.
pub fn trace(self: Self) f64 {
    assert(self.data.len == self.data[0].len);
    var sum: f64 = 0;
    for (0..self.data.len) |i| {
        sum += self.data[i][i];
    }
    return sum;
}

test "Matrix.trace" {
    try std.testing.expectEqual(@as(f64, 15), X.trace());
}

/// Overwrites the matrix by adding a number to every entry.
pub fn addScalar(self: Self, scalar: f64) void {
    for (self.data) |col| {
        for (col) |*entry| {
            entry.* += scalar;
        }
    }
}

test "Matrix.addScalar" {
    const Y = try X.dupe();
    defer Y.destroy();
    Y.addScalar(3);
    try std.testing.expectEqualSlices(f64, &.{4 , 5 , 6 }, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{7 , 8 , 9 }, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{10, 11, 12}, Y.data[2]);
}

/// Overwrites the matrix by multiplying a number to every entry.
pub fn multiplyScalar(self: Self, scalar: f64) void {
    for (self.data) |col| {
        for (col) |*entry| {
            entry.* *= scalar;
        }
    }
}

test "Matrix.multiplyScalar" {
    const Y = try X.dupe();
    defer Y.destroy();
    Y.multiplyScalar(3);
    try std.testing.expectEqualSlices(f64, &.{3 , 6 , 9 }, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{12, 15, 18}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{21, 24, 27}, Y.data[2]);
}

/// Overwrites the matrix by adding another one to it,
/// they must have the same amount of rows and columns.
pub fn addMatrix(self: Self, rhs: Self) void {
    assert(self.data.len == rhs.data.len);
    assert(self.data[0].len == rhs.data[0].len);
    for (self.data, rhs.data) |col1, col2| {
        for (col1, col2) |*entry1, entry2| {
            entry1.* += entry2;
        }
    }
}

test "Matrix.addMatrix" {
    const Y = try X.dupe();
    defer Y.destroy();
    Y.addMatrix(X);
    try std.testing.expectEqualSlices(f64, &.{2 , 4 , 6 }, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{8 , 10, 12}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{14, 16, 18}, Y.data[2]);
}

/// Creates a new matrix by doing matrix multiplication between lhs and rhs
/// number of columns of lhs must equal number of rows of rhs,
/// result must be freed by the caller with `destroy`.
pub fn multiplyMatrix(self: Self, rhs: Self) !Self {
    assert(self.data.len == rhs.data[0].len);
    const rows = self.data[0].len;
    const cols = rhs.data.len;
    const result = try self.create(rows, cols);
    for (0..rows) |j| {
        for (0..cols) |i| {
            result.data[j][i] = 0;
            for (0..self.data.len) |k| {
                result.data[j][i] += self.data[k][i] * rhs.data[j][k];
            }
        }
    }
    return result;
}

test "Matrix.multiplyMatrix" {
    const Y = try X.multiplyMatrix(X);
    defer Y.destroy();
    try std.testing.expectEqualSlices(f64, &.{30 , 36 , 42 }, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{66 , 81 , 96 }, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{102, 126, 150}, Y.data[2]);
}

/// Calculates the determinant of a square matrix.
pub fn determinant(self: Self) !f64 {
    assert(self.data.len == self.data[0].len);
    const result = try self.dupe();
    defer result.destroy();
    const n = result.data.len;
    for (0..n - 1) |k| {
        for (k + 1..n) |j| {
            for (k + 1..n) |i| {
                const num1 = result.data[j][i] * result.data[k][k];
                const num2 = result.data[k][i] * result.data[j][k];
                const den = if (k == 0) 1 else result.data[k - 1][k - 1];
                result.data[j][i] = (num1 - num2) / den;
            }
        }
    }
    return result.data[n - 1][n - 1];
}

test "Matrix.determinant" {
    const Y = try X.dupe();
    defer Y.destroy();
    Y.data[1][1] = 55;
    const det = try Y.determinant();
    try std.testing.expectEqual(@as(f64, -600), det);
}

/// Creates a new matrix by removing a specified row and column by its index,
/// result must be freed by the caller with `destroy`.
pub fn minor(self: Self, row: usize, col: usize) !Self {
    const rows = self.data[0].len - 1;
    const cols = self.data.len - 1;
    const result = try self.create(rows, cols);
    for (0..cols) |j| {
        const skippedcol = j + @intFromBool(j >= col);
        for (0..rows) |i| {
            const skippedrow = i + @intFromBool(i >= row);
            result.data[j][i] = self.data[skippedcol][skippedrow];
        }
    }
    return result;
}

test "Matrix.minor" {
    const Y = try X.minor(1, 1);
    defer Y.destroy();
    try std.testing.expectEqualSlices(f64, &.{1, 3}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{7, 9}, Y.data[1]);
}

/// Creates a new matrix that is the inverse of a square matrix,
/// result must be freed by the caller with `destroy`.
pub fn inverse(self: Self) !Self {
    const det = try self.determinant();
    const result = try self.dupe();
    for (0..result.data.len) |j| {
        for (0..result.data.len) |i| {
            const minor_matrix = try self.minor(i, j);
            defer minor_matrix.destroy();
            const cofactor = try minor_matrix.determinant();
            const sign: f64 = if ((i + j) % 2 == 0) 1 else -1;
            result.data[i][j] = sign * cofactor / det;
        }
    }
    return result;
}

test "Matrix.inverse" {
    const Y = try X.dupe();
    defer Y.destroy();
    Y.data[1][1] = 55;
    const Z = try Y.inverse();
    defer Z.destroy();
    Z.multiplyScalar(1 / try Z.determinant());
    try std.testing.expectEqualSlices(f64, &.{ 447,   6, -153}, Z.data[0]);
    try std.testing.expectEqualSlices(f64, &.{   6, -12,    6}, Z.data[1]);
    try std.testing.expectEqualSlices(f64, &.{-353,   6,   47}, Z.data[2]);
}
