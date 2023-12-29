

const std = @import("std");
const Matrix = @import("Matrix.zig");

pub const Config = struct {
    delimiter_vertical: u8 = '\n',
    delimiter_horizontal: u8 = ',',
};

pub fn read(allocator: std.mem.Allocator, path: []const u8, comptime config: Config) !Matrix {
    const file = try std.fs.cwd().readFileAlloc(
        allocator,
        path,
        std.math.maxInt(usize),
    );
    defer allocator.free(file);

    var iterator_vertical = std.mem
        .splitScalar(u8, file, config.delimiter_vertical);
    const colnames_line = iterator_vertical.first();
    const cols = std.mem.count(u8, colnames_line, &.{config.delimiter_horizontal}) + 1;
    const rows = std.mem.count(u8, file, &.{config.delimiter_vertical}) - 1;

    const result = try Matrix
        .init(allocator)
        .create(rows, cols);
    errdefer result.destroy();

    var i: usize = 0;
    while (iterator_vertical.next()) |line| : (i += 1) {
        if (i == result.data[0].len) {
            continue;
        }
        var iterator_horizontal = std.mem
            .splitScalar(u8, line, config.delimiter_horizontal);
        var j: usize = 0;
        while (iterator_horizontal.next()) |value| : (j += 1) {
            result.data[j][i] = try std.fmt.parseFloat(f64, value);
        }
    }
    return result;
}
