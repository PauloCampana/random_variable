const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn create(allocator: Allocator, rows: usize, columns: usize) ![][]f64 {
    const matrix = try allocator.alloc([]f64, columns);
    for (matrix) |*col| {
        col.* = try allocator.alloc(f64, rows);
    }
    return matrix;
}

pub fn createIdentity(allocator: Allocator, n: usize) ![][]f64 {
    const result = try create(allocator, n, n);
    for (result, 0..) |col, i| {
        @memset(col, 0);
        col[i] = 1;
    }
    return result;
}

pub fn createDiagonal(allocator: Allocator, slice: []f64) ![][]f64 {
    const result = try create(allocator, slice.len, slice.len);
    for (result, 0..) |col, i| {
        @memset(col, 0);
        col[i] = slice[i];
    }
    return result;
}

pub fn destroy(allocator: Allocator, matrix: [][]f64) void {
    for (matrix) |col| {
        allocator.free(col);
    }
    allocator.free(matrix);
}

pub fn dupe(allocator: Allocator, matrix: [][]f64) ![][]f64 {
    const result = try allocator.dupe([]f64, matrix);
    for (matrix, result) |old, *new| {
        new.* = try allocator.dupe(f64, old);
    }
    return result;
}

pub fn getDiagonal(allocator: Allocator, matrix: [][]f64) ![]f64 {
    const result = try allocator.alloc(f64, matrix.len);
    for (result, 0..) |*entry, i| {
        entry.* = matrix[i][i];
    }
    return result;
}

pub fn transpose(allocator: Allocator, matrix: [][]f64) ![][]f64 {
    const result = try create(allocator, matrix.len, matrix[0].len);
    for (result, 0..) |col, j| {
        for (col, 0..) |*entry, i| {
            entry.* = matrix[i][j];
        }
    }
    return result;
}

pub fn trace(matrix: [][]f64) f64 {
    var sum: f64 = 0;
    for (matrix, 0..) |col, i| {
        sum += col[i];
    }
    return sum;
}

pub fn addScalar(allocator: Allocator, matrix: [][]f64, scalar: f64) ![][]f64 {
    const result = try dupe(allocator, matrix);
    return addScalarInPlace(result, scalar);
}

pub fn addScalarInPlace(matrix: [][]f64, scalar: f64) [][]f64 {
    for (matrix) |col| {
        for (col) |*entry| {
            entry.* += scalar;
        }
    }
    return matrix;
}

pub fn addMatrix(allocator: Allocator, lhs: [][]f64, rhs: [][]f64) ![][]f64 {
    const result = try dupe(allocator, lhs);
    return addMatrixInPlace(result, rhs);
}

pub fn addMatrixInPlace(lhs: [][]f64, rhs: [][]f64) [][]f64 {
    for (lhs, rhs) |col1, col2| {
        for (col1, col2) |*entry1, entry2| {
            entry1.* += entry2;
        }
    }
    return lhs;
}

pub fn multiplyScalar(allocator: Allocator, matrix: [][]f64, scalar: f64) ![][]f64 {
    const result = try dupe(allocator, matrix);
    return multiplyScalarInPlace(result, scalar);
}

pub fn multiplyScalarInPlace(matrix: [][]f64, scalar: f64) [][]f64 {
    for (matrix) |col| {
        for (col) |*entry| {
            entry.* *= scalar;
        }
    }
    return matrix;
}

pub fn multiplyMatrix(allocator: Allocator, lhs: [][]f64, rhs: [][]f64) ![][]f64 {
    const result = try create(allocator, lhs[0].len, rhs.len);
    for (result, 0..) |col, j| {
        for (col, 0..) |*entry, i| {
            entry.* = 0;
            for (0..lhs.len) |k| {
                entry.* += lhs[k][i] * rhs[j][k];
            }
        }
    }
    return result;
}

pub fn determinant(allocator: Allocator, matrix: [][]f64) !f64 {
    const result = try dupe(allocator, matrix);
    defer destroy(allocator, result);
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

pub fn minor(allocator: Allocator, matrix: [][]f64, row: usize, column: usize) ![][]f64 {
    const result = try create(allocator, matrix[0].len - 1, matrix.len - 1);
    for (result, 0..) |col, j| {
        const skippedj = if (j < column) j else j + 1;
        for (col, 0..) |*entry, i| {
            const skippedi = if (i < row) i else i + 1;
            entry.* = matrix[skippedj][skippedi];
        }
    }
    return result;
}

pub fn inverse(allocator: Allocator, matrix: [][]f64) ![][]f64 {
    const det = try determinant(allocator, matrix);
    const result = try dupe(allocator, matrix);
    for (0..result.len) |j| {
        for (0..result.len) |i| {
            const minor_matrix = try minor(allocator, matrix, i, j);
            defer destroy(allocator, minor_matrix);
            const cofactor = try determinant(allocator, minor_matrix);
            const sign: f64 = if ((i + j) % 2 == 0) 1 else -1;
            result[i][j] = sign * cofactor / det;
        }
    }
    return result;
}
