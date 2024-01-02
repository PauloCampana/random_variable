//! Dynamically allocated matrices,
//! start by using `init()` with an allocator.

const std = @import("std");
const Self = @This();

allocator: std.mem.Allocator,
data: [][]f64,

/// Sets the allocator to be used for further functions.
pub fn init(allocator: std.mem.Allocator) Self {
    return Self {
        .allocator = allocator,
        .data = undefined,
    };
}

/// Function that gets called by `std.fmt.format` whenever
/// you use a `print` function from a writer.
pub fn format(
    self: Self,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    _ = options;
    const new_fmt = if (fmt.len == 0) "{d:10.3}" else fmt;
    const rows = self.data[0].len;
    const cols = self.data.len;
    try writer.print("[{}][{}]Matrix {{", .{rows, cols});
    for (0..@min(rows, 10)) |i| {
        try writer.print("\n    ", .{});
        for (0..@min(cols, 10)) |j| {
            try writer.print(new_fmt, .{self.data[j][i]});
        }
    }
    try writer.print("\n}}\n", .{});
}

/// Allocates memory for a rows×columns matrix,
/// the memory is undefined and must first be written to before reading,
/// result must be freed by the caller with `free()`.
pub fn alloc(self: Self, rows: usize, cols: usize) !Self {
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
pub fn free(self: Self) void {
    for (self.data) |col| {
        self.allocator.free(col);
    }
    self.allocator.free(self.data);
}

/// Creates a new matrix with the contents of another,
/// result must be freed by the caller with `free()`.
pub fn dupe(self: Self) !Self {
    const result = try self.allocator.dupe([]f64, self.data);
    for (result, self.data) |*r, x| {
        r.* = try self.allocator.dupe(f64, x);
    }
    return Self {
        .allocator = self.allocator,
        .data = result,
    };
}

/// Uses the values of a slice to create a new matrix sequentially by columns,
/// errors when `slice.len` isn't divisible by `cols` or when `cols` is 0,
/// result must be freed by the caller with `free()`.
pub fn createFromSlice(self: Self, slice: []const f64, cols: usize) !Self {
    const rows = try std.math.divExact(usize, slice.len, cols);
    const result = try self.alloc(rows, cols);
    for (result.data, 0..) |col, j| {
        @memcpy(col, slice[j * rows..][0..rows]);
    }
    return result;
}

/// Can be used with literals such as `&.{&.{1, 2}, &.{3, 4}}`,
/// result must be freed by the caller with `free()`.
pub fn createFromSliceOfSlices(self: Self, sos: []const []const f64) !Self {
    const rows = sos[0].len;
    const cols = sos.len;
    const result = try self.alloc(rows, cols);
    for (result.data, sos) |col, slice| {
        @memcpy(col, slice);
    }
    return result;
}

/// Creates a new n×n square matrix where
/// entries in the main diagonal are 1, otherwise 0,
/// result must be freed by the caller with `free()`.
pub fn createIdentity(self: Self, n: usize) !Self {
    const result = try self.alloc(n, n);
    for (result.data, 0..) |col, i| {
        @memset(col, 0);
        col[i] = 1;
    }
    return result;
}

/// Creates a new n×n square matrix where
/// entries in the main diagonal are taken from a slice,
/// result must be freed by the caller with `free()`.
pub fn createDiagonal(self: Self, slice: []const f64) !Self {
    const n = slice.len;
    const result = try self.alloc(n, n);
    for (result.data, 0..) |col, i| {
        @memset(col, 0);
        col[i] = slice[i];
    }
    return result;
}

/// Returns the entries in the main diagonal of a square as a slice,
/// result must be freed by the caller.
pub fn getDiagonal(self: Self) ![]f64 {
    if (self.data.len != self.data[0].len) {
        return error.MatrixNotSquare;
    }
    const n = self.data.len;
    const result = try self.allocator.alloc(f64, n);
    for (result, 0..) |*r, i| {
        r.* = self.data[i][i];
    }
    return result;
}

/// Creates a new flipped version of a matrix over its main diagonal,
/// the transpose of a matrix of size a×b is size b×a,
/// result must be freed by the caller with `free()`.
pub fn transpose(self: Self) !Self {
    const new_rows = self.data.len;
    const new_cols = self.data[0].len;
    const result = try self.alloc(new_rows, new_cols);
    for (0..new_cols) |j| {
        for (0..new_rows) |i| {
            result.data[j][i] = self.data[i][j];
        }
    }
    return result;
}

/// Calculates the sum of a square matrix's main diagonal entries.
pub fn trace(self: Self) f64 {
    std.debug.assert(self.data.len == self.data[0].len);
    var sum: f64 = 0;
    for (0..self.data.len) |i| {
        sum += self.data[i][i];
    }
    return sum;
}

/// Overwrites the matrix by adding a number to every entry.
pub fn addScalar(self: Self, scalar: f64) void {
    for (self.data) |col| {
        for (col) |*x| {
            x.* += scalar;
        }
    }
}

/// Overwrites the matrix by multiplying a number to every entry.
pub fn multiplyScalar(self: Self, scalar: f64) void {
    for (self.data) |col| {
        for (col) |*x| {
            x.* *= scalar;
        }
    }
}

/// Overwrites the matrix by adding elementwise
/// another matrix of the same size.
pub fn addMatrix(self: Self, rhs: Self) void {
    std.debug.assert(self.data.len == rhs.data.len);
    std.debug.assert(self.data[0].len == rhs.data[0].len);
    for (self.data, rhs.data) |col1, col2| {
        for (col1, col2) |*y, x| {
            y.* += x;
        }
    }
}

/// Overwrites the matrix by subtracting elementwise
/// another matrix of the same size.
pub fn subtractMatrix(self: Self, rhs: Self) void {
    std.debug.assert(self.data.len == rhs.data.len);
    std.debug.assert(self.data[0].len == rhs.data[0].len);
    for (self.data, rhs.data) |col1, col2| {
        for (col1, col2) |*y, x| {
            y.* -= x;
        }
    }
}

/// Creates a new matrix by matrix multiplication,
/// when multiplying matrices of size a×b and c×d,
/// b and c must be equal and the result is size a×d,
/// result must be freed by the caller with `free()`.
pub fn multiplyMatrix(self: Self, rhs: Self) !Self {
    if (self.data.len != rhs.data[0].len) {
        return error.InvalidSizesForMultiplication;
    }
    const rows = self.data[0].len;
    const cols = rhs.data.len;
    const result = try self.alloc(rows, cols);
    for (0..cols) |j| {
        for (0..rows) |i| {
            result.data[j][i] = 0;
            for (0..self.data.len) |k| {
                result.data[j][i] += self.data[k][i] * rhs.data[j][k];
            }
        }
    }
    return result;
}

/// Calculates the determinant of a square matrix.
pub fn determinant(self: Self) !f64 {
    if (self.data.len != self.data[0].len) {
        return error.MatrixNotSquare;
    }
    const result = try self.dupe();
    defer result.free();
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

/// Creates a new matrix by removing a specified row and column by its index,
/// result must be freed by the caller with `free()`.
pub fn minor(self: Self, row: usize, col: usize) !Self {
    const rows = self.data[0].len - 1;
    const cols = self.data.len - 1;
    const result = try self.alloc(rows, cols);
    for (0..cols) |j| {
        const skippedj = j + @intFromBool(j >= col);
        for (0..rows) |i| {
            const skippedi = i + @intFromBool(i >= row);
            result.data[j][i] = self.data[skippedj][skippedi];
        }
    }
    return result;
}

/// Creates a new matrix that is the inverse of a square matrix,
/// errors when the matrix is not invertible,
/// result must be freed by the caller with `free()`.
pub fn inverse(self: Self) !Self {
    const det = try self.determinant();
    if (det == 0) {
        return error.MatrixNotInvertible;
    }
    const result = try self.dupe();
    for (0..result.data.len) |j| {
        for (0..result.data.len) |i| {
            const minor_matrix = try self.minor(i, j);
            defer minor_matrix.free();
            const cofactor = try minor_matrix.determinant();
            const sign: f64 = if ((i + j) % 2 == 0) 1 else -1;
            result.data[i][j] = sign * cofactor / det;
        }
    }
    return result;
}

const Matrix = Self.init(std.testing.allocator);

test "Matrix.dupe" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    const Y = try X.dupe();
    defer Y.free();
    Y.data[1][1] = 55;
    try std.testing.expectEqual(@as(f64, 55), Y.data[1][1]);
    try std.testing.expectEqual(@as(f64, 5 ), X.data[1][1]);
}

test "Matrix.createFromSliceOfSlices" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    const Y = try Matrix.createFromSliceOfSlices(&.{
        &.{1, 2, 3},
        &.{4, 5, 6},
        &.{7, 8, 9},
    });
    defer Y.free();
    try std.testing.expectEqualSlices(f64, X.data[0], Y.data[0]);
    try std.testing.expectEqualSlices(f64, X.data[1], Y.data[1]);
    try std.testing.expectEqualSlices(f64, X.data[2], Y.data[2]);
}

test "Matrix.createIdentity" {
    const Y = try Matrix.createIdentity(3);
    defer Y.free();
    try std.testing.expectEqualSlices(f64, &.{1, 0, 0}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{0, 1, 0}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{0, 0, 1}, Y.data[2]);
}

test "Matrix.createDiagonal" {
    const Y = try Matrix.createDiagonal(&.{1, 2, 3});
    defer Y.free();
    try std.testing.expectEqualSlices(f64, &.{1, 0, 0}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{0, 2, 0}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{0, 0, 3}, Y.data[2]);
}

test "Matrix.getDiagonal" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    const slice = try X.getDiagonal();
    defer X.allocator.free(slice);
    try std.testing.expectEqualSlices(f64, &.{1, 5, 9}, slice);
}

test "Matrix.transpose" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    const Y = try X.transpose();
    defer Y.free();
    try std.testing.expectEqualSlices(f64, &.{1, 4, 7}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{2, 5, 8}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{3, 6, 9}, Y.data[2]);
}

test "Matrix.trace" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    try std.testing.expectEqual(@as(f64, 15), X.trace());
}

test "Matrix.addScalar" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    X.addScalar(3);
    try std.testing.expectEqualSlices(f64, &.{4 , 5 , 6 }, X.data[0]);
    try std.testing.expectEqualSlices(f64, &.{7 , 8 , 9 }, X.data[1]);
    try std.testing.expectEqualSlices(f64, &.{10, 11, 12}, X.data[2]);
}

test "Matrix.multiplyScalar" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    X.multiplyScalar(3);
    try std.testing.expectEqualSlices(f64, &.{3 , 6 , 9 }, X.data[0]);
    try std.testing.expectEqualSlices(f64, &.{12, 15, 18}, X.data[1]);
    try std.testing.expectEqualSlices(f64, &.{21, 24, 27}, X.data[2]);
}

test "Matrix.addMatrix" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    X.addMatrix(X);
    try std.testing.expectEqualSlices(f64, &.{2 , 4 , 6 }, X.data[0]);
    try std.testing.expectEqualSlices(f64, &.{8 , 10, 12}, X.data[1]);
    try std.testing.expectEqualSlices(f64, &.{14, 16, 18}, X.data[2]);
}

test "Matrix.subtractMatrix" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    X.subtractMatrix(X);
    try std.testing.expectEqualSlices(f64, &.{0, 0, 0}, X.data[0]);
    try std.testing.expectEqualSlices(f64, &.{0, 0, 0}, X.data[1]);
    try std.testing.expectEqualSlices(f64, &.{0, 0, 0}, X.data[2]);
}

test "Matrix.multiplyMatrix" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    const Y = try X.multiplyMatrix(X);
    defer Y.free();
    try std.testing.expectEqualSlices(f64, &.{30 , 36 , 42 }, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{66 , 81 , 96 }, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{102, 126, 150}, Y.data[2]);
}

test "Matrix.determinant" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 55, 6, 7, 8, 9}, 3);
    defer X.free();
    const det = try X.determinant();
    try std.testing.expectEqual(@as(f64, -600), det);
}

test "Matrix.minor" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);
    defer X.free();
    const Y = try X.minor(1, 1);
    defer Y.free();
    try std.testing.expectEqualSlices(f64, &.{1, 3}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{7, 9}, Y.data[1]);
}

test "Matrix.inverse" {
    const X = try Matrix.createFromSlice(&.{1, 2, 3, 4, 55, 6, 7, 8, 9}, 3);
    defer X.free();
    const Y = try X.inverse();
    defer Y.free();
    Y.multiplyScalar(1 / try Y.determinant());
    try std.testing.expectEqualSlices(f64, &.{ 447,   6, -153}, Y.data[0]);
    try std.testing.expectEqualSlices(f64, &.{   6, -12,    6}, Y.data[1]);
    try std.testing.expectEqualSlices(f64, &.{-353,   6,   47}, Y.data[2]);
}
