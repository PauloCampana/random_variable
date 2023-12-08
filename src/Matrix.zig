//! Functions for manipulation of dynamically allocated matrices

const std = @import("std");
const assert = std.debug.assert;
const Self = @This();

// for testing purposes
const Matrix = Self.setAllocator(std.testing.allocator);
var first = [_]f64 {1, 2, 3};
var secon = [_]f64 {4, 5, 6};
var third = [_]f64 {7, 8, 9};
var all = [_][]f64 {&first, &secon, &third};
const X: [][]f64 = &all;

allocator: std.mem.Allocator,

/// Specify an allocator for the returning matrices and slices.
pub fn setAllocator(allocator: std.mem.Allocator) Self {
    return Self {.allocator = allocator};
}

/// Prints a matrix in the correct orientation,
/// output is customizable with `fmt`, such as "{d:.3}, ".
pub fn print(self: Self, comptime fmt: []const u8, matrix: [][]f64) void {
    _ = self;
    const printf = std.debug.print;
    const rows = matrix[0].len;
    const cols = matrix.len;
    printf("[{} x {}] {{\n", .{rows, cols});
    for (0..rows) |i| {
        printf("    ", .{});
        for (0..cols) |j| {
            printf(fmt, .{matrix[j][i]});
        }
        printf("\n", .{});
    }
    printf("}}\n", .{});
} // TODO change from debug.print to io.Writer

/// Allocates memory for a rows×columns matrix,
/// the memory is undefined and must first be written before reading,
/// result must be freed by the caller with `destroy`.
pub fn create(self: Self, rows: usize, columns: usize) ![][]f64 {
    const result = try self.allocator.alloc([]f64, columns);
    for (result) |*col| {
        col.* = try self.allocator.alloc(f64, rows);
    }
    return result;
}

/// Frees the allocated memory of a matrix.
pub fn destroy(self: Self, matrix: [][]f64) void {
    for (matrix) |col| {
        self.allocator.free(col);
    }
    self.allocator.free(matrix);
}

test "Matrix.create, Matrix.destroy" {
    const Y = try Matrix.create(3, 3);
    defer Matrix.destroy(Y);
    for (Y, X) |coly, colx| {
        for (coly, colx) |*y, x| {
            y.* = x;
        }
    }
    try std.testing.expectEqualSlices(f64, &.{1,2,3}, Y[0]);
    try std.testing.expectEqualSlices(f64, &.{4,5,6}, Y[1]);
    try std.testing.expectEqualSlices(f64, &.{7,8,9}, Y[2]);
}

/// Creates a new matrix with the contents of another,
/// result must be freed by the caller with `destroy`.
pub fn dupe(self: Self, matrix: [][]f64) ![][]f64 {
    const result = try self.allocator.dupe([]f64, matrix);
    for (matrix, result) |old, *new| {
        new.* = try self.allocator.dupe(f64, old);
    }
    return result;
}

test "Matrix.dupe" {
    const Y = try Matrix.dupe(X);
    defer Matrix.destroy(Y);
    Y[1][1] = 55;
    try std.testing.expectEqual(@as(f64, 55), Y[1][1]);
    try std.testing.expectEqual(@as(f64, 5 ), X[1][1]);
}

/// Creates a new n×n identity matrix,
/// result must be freed by the caller with `destroy`.
pub fn createIdentity(self: Self, n: usize) ![][]f64 {
    const result = try self.create(n, n);
    for (0..n) |i| {
        @memset(result[i], 0);
        result[i][i] = 1;
    }
    return result;
}

test "Matrix.createIdentity" {
    const I = try Matrix.createIdentity(3);
    defer Matrix.destroy(I);
    try std.testing.expectEqualSlices(f64, &.{1,0,0}, I[0]);
    try std.testing.expectEqualSlices(f64, &.{0,1,0}, I[1]);
    try std.testing.expectEqualSlices(f64, &.{0,0,1}, I[2]);
}

/// Creates a new square matrix where the diagonal is taken from a slice,
/// result must be freed by the caller with `destroy`.
pub fn createDiagonal(self: Self, slice: []f64) ![][]f64 {
    const n = slice.len;
    const result = try self.create(n, n);
    for (0..n) |i| {
        @memset(result[i], 0);
        result[i][i] = slice[i];
    }
    return result;
}

test "Matrix.createDiagonal" {
    var array = [_]f64 {1, 2, 3};
    const D = try Matrix.createDiagonal(&array);
    defer Matrix.destroy(D);
    try std.testing.expectEqualSlices(f64, &.{1,0,0}, D[0]);
    try std.testing.expectEqualSlices(f64, &.{0,2,0}, D[1]);
    try std.testing.expectEqualSlices(f64, &.{0,0,3}, D[2]);
}

/// Returns the diagonal entries of the matrix as a slice,
/// result must be freed by the caller.
pub fn getDiagonal(self: Self, matrix: [][]f64) ![]f64 {
    assert(matrix.len == matrix[0].len);
    const n = matrix.len;
    const result = try self.allocator.alloc(f64, n);
    for (0..n) |i| {
        result[i] = matrix[i][i];
    }
    return result;
}

test "Matrix.getDiagonal" {
    const slice = try Matrix.getDiagonal(X);
    defer Matrix.allocator.free(slice);
    try std.testing.expectEqualSlices(f64, &.{1,5,9}, slice);
}

/// Creates a new flipped version of a matrix over its main diagonal,
/// the transpose of a matrix of size a×b is size b×a,
/// result must be freed by the caller with `destroy`.
pub fn transpose(self: Self, matrix: [][]f64) ![][]f64 {
    const rows = matrix[0].len;
    const cols = matrix.len;
    const result = try self.create(cols, rows);
    for (0..cols) |j| {
        for (0..rows) |i| {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

test "Matrix.transpose" {
    const T = try Matrix.transpose(X);
    defer Matrix.destroy(T);
    try std.testing.expectEqualSlices(f64, &.{1,4,7}, T[0]);
    try std.testing.expectEqualSlices(f64, &.{2,5,8}, T[1]);
    try std.testing.expectEqualSlices(f64, &.{3,6,9}, T[2]);
}

/// Calculates the sum of the diagonal entries of a square matrix.
pub fn trace(matrix: [][]f64) f64 {
    assert(matrix.len == matrix[0].len);
    var sum: f64 = 0;
    for (0..matrix.len) |i| {
        sum += matrix[i][i];
    }
    return sum;
}

test "Matrix.trace" {
    try std.testing.expectEqual(@as(f64, 15), trace(X));
}

/// Overwrites a matrix by adding a number to every entry.
pub fn addScalar(self: Self, matrix: [][]f64, scalar: f64) void {
    _ = self;
    for (matrix) |col| {
        for (col) |*entry| {
            entry.* += scalar;
        }
    }
}

test "Matrix.addScalar" {
    const Y = try Matrix.dupe(X);
    defer Matrix.destroy(Y);
    Matrix.addScalar(Y, 3);
    try std.testing.expectEqualSlices(f64, &.{4 ,5 ,6 }, Y[0]);
    try std.testing.expectEqualSlices(f64, &.{7 ,8 ,9 }, Y[1]);
    try std.testing.expectEqualSlices(f64, &.{10,11,12}, Y[2]);
}

/// Overwrites the lhs matrix by adding the rhs matrix to it,
/// they must have the same amount of rows and columns.
pub fn addMatrix(self: Self, lhs: [][]f64, rhs: [][]f64) void {
    _ = self;
    assert(lhs.len == rhs.len);
    assert(lhs[0].len == rhs[0].len);
    for (lhs, rhs) |col1, col2| {
        for (col1, col2) |*entry1, entry2| {
            entry1.* += entry2;
        }
    }
}

test "Matrix.addMatrix" {
    const Y = try Matrix.dupe(X);
    defer Matrix.destroy(Y);
    Matrix.addMatrix(Y, X);
    try std.testing.expectEqualSlices(f64, &.{2 ,4 ,6 }, Y[0]);
    try std.testing.expectEqualSlices(f64, &.{8 ,10,12}, Y[1]);
    try std.testing.expectEqualSlices(f64, &.{14,16,18}, Y[2]);
}

/// Overwrites a matrix by multiplying a number to every entry.
pub fn multiplyScalar(self: Self, matrix: [][]f64, scalar: f64) void {
    _ = self;
    for (matrix) |col| {
        for (col) |*entry| {
            entry.* *= scalar;
        }
    }
}

test "Matrix.multiplyScalar" {
    const Y = try Matrix.dupe(X);
    defer Matrix.destroy(Y);
    Matrix.multiplyScalar(Y, 3);
    try std.testing.expectEqualSlices(f64, &.{3 ,6 ,9 }, Y[0]);
    try std.testing.expectEqualSlices(f64, &.{12,15,18}, Y[1]);
    try std.testing.expectEqualSlices(f64, &.{21,24,27}, Y[2]);
}

/// Creates a new matrix by doing matrix multiplication between lhs and rhs
/// number of columns of lhs must equal number of rows of rhs,
/// result must be freed by the caller with `destroy`.
pub fn multiplyMatrix(self: Self, lhs: [][]f64, rhs: [][]f64) ![][]f64 {
    assert(lhs.len == rhs[0].len);
    const rows = lhs[0].len;
    const cols = rhs.len;
    const result = try self.create(lhs[0].len, rhs.len);
    for (0..rows) |j| {
        for (0..cols) |i| {
            result[j][i] = 0;
            for (0..lhs.len) |k| {
                result[j][i] += lhs[k][i] * rhs[j][k];
            }
        }
    }
    return result;
}

test "Matrix.multiplyMatrix" {
    const Y = try Matrix.multiplyMatrix(X, X);
    defer Matrix.destroy(Y);
    try std.testing.expectEqualSlices(f64, &.{30 ,36 ,42 }, Y[0]);
    try std.testing.expectEqualSlices(f64, &.{66 ,81 ,96 }, Y[1]);
    try std.testing.expectEqualSlices(f64, &.{102,126,150}, Y[2]);
}

/// Calculates the determinant of a square matrix.
pub fn determinant(self: Self, matrix: [][]f64) !f64 {
    assert(matrix.len == matrix[0].len);
    const result = try self.dupe(matrix);
    defer self.destroy(result);
    const n = result.len;
    for (0..n - 1) |k| {
        for (k + 1..n) |j| {
            for (k + 1..n) |i| {
                const num1 = result[j][i] * result[k][k];
                const num2 = result[k][i] * result[j][k];
                const den = if (k == 0) 1 else result[k - 1][k - 1];
                result[j][i] = (num1 - num2) / den;
            }
        }
    }
    return result[n - 1][n - 1];
}

test "Matrix.determinant" {
    const Y = try Matrix.dupe(X);
    defer Matrix.destroy(Y);
    Y[1][1] = 55;
    const det = try Matrix.determinant(Y);
    try std.testing.expectEqual(@as(f64, -600), det);
}

/// Creates a new matrix by removing a specified row and column by its index,
/// result must be freed by the caller with `destroy`.
pub fn minor(self: Self, matrix: [][]f64, row: usize, column: usize) ![][]f64 {
    const rows = matrix[0].len - 1;
    const cols = matrix.len - 1;
    const result = try self.create(rows, cols);
    for (0..cols) |j| {
        const skippedcol = j + @intFromBool(j >= column);
        for (0..rows) |i| {
            const skippedrow = i + @intFromBool(i >= row);
            result[j][i] = matrix[skippedcol][skippedrow];
        }
    }
    return result;
}

test "Matrix.minor" {
    const Y = try Matrix.minor(X, 1, 1);
    defer Matrix.destroy(Y);
    try std.testing.expectEqualSlices(f64, &.{1,3}, Y[0]);
    try std.testing.expectEqualSlices(f64, &.{7,9}, Y[1]);
}

/// Creates a new matrix that is the inverse of a square matrix,
/// result must be freed by the caller with `destroy`.
pub fn inverse(self: Self, matrix: [][]f64) ![][]f64 {
    const det = try self.determinant(matrix);
    const result = try self.dupe(matrix);
    for (0..result.len) |j| {
        for (0..result.len) |i| {
            const minor_matrix = try self.minor(matrix, i, j);
            defer self.destroy(minor_matrix);
            const cofactor = try self.determinant(minor_matrix);
            const sign: f64 = if ((i + j) % 2 == 0) 1 else -1;
            result[i][j] = sign * cofactor / det;
        }
    }
    return result;
}

test "Matrix.inverse" {
    const Y = try Matrix.dupe(X);
    defer Matrix.destroy(Y);
    Y[1][1] = 55;
    const Z = try Matrix.inverse(Y);
    defer Matrix.destroy(Z);
    Matrix.multiplyScalar(Z, 1 / try Matrix.determinant(Z));
    try std.testing.expectEqualSlices(f64, &.{ 447,   6, -153}, Z[0]);
    try std.testing.expectEqualSlices(f64, &.{   6, -12,    6}, Z[1]);
    try std.testing.expectEqualSlices(f64, &.{-353,   6,   47}, Z[2]);
}
