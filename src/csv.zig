const std = @import("std");
const Matrix = @import("Matrix.zig");

pub const Config = struct {
    delimiter_vertical: u8 = '\n',
    delimiter_horizontal: u8 = ',',
    Enums: []const type,
};

pub fn read(allocator: std.mem.Allocator, path: []const u8, comptime config: Config) !Matrix {
    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();
    const all = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(all);

    var iterator_vertical = std.mem
        .splitScalar(u8, all, config.delimiter_vertical);
    const colnames_line = iterator_vertical.first();
    const cols = std.mem.count(u8, colnames_line, &.{config.delimiter_horizontal}) + 1;
    const rows = std.mem.count(u8, all, &.{config.delimiter_vertical}) - 1;
        const result = try Matrix
        .setAllocator(allocator)
        .create(rows, cols);

    var i: usize = 0;
    while (iterator_vertical.next()) |line| : (i += 1) {
        if (i == result.data[0].len) {
            continue;
        }
        var iterator_horizontal = std.mem
            .splitScalar(u8, line, config.delimiter_horizontal);
        var j: usize = 0;
        while (iterator_horizontal.next()) |value| : (j += 1) {
            result.data[j][i] = std.fmt.parseFloat(f64, value) catch blk: {
                inline for (config.Enums) |Enum| {
                    if (std.meta.stringToEnum(Enum, value)) |x| {
                        break :blk @floatFromInt(@intFromEnum(x));
                    }
                }
                @panic("Found a string which does not match the specified enum for that column");
            };
        }
    }
    return result;
}

test "csv.read" {
    const DiamondsCut = enum {
        Fair, Good, @"Very Good", Premium, Ideal,
    };
    const DiamondsColor = enum {
        D, E, F, G, H, I, J,
    };
    const DiamondsClarity = enum {
        I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF,
    };

    const diamonds = try read(
        std.testing.allocator,
        "/home/paulo/Documents/R/data/diamonds.csv",
        .{.Enums = &[_]type {DiamondsCut, DiamondsColor, DiamondsClarity}}
    );
    try diamonds.print(std.io.getStdErr().writer(), "{d:8}, ");
    defer diamonds.destroy();
}
