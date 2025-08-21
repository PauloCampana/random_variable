const std = @import("std");
const builtin = @import("builtin");

const Count = struct {
    success: u32 = 0,
    fail: u32 = 0,
};

pub fn main() !void {
    const test_fns = builtin.test_functions;
    var count: Count = .{};

    var timer = try std.time.Timer.start();
    const progress = std.Progress.start(.{
        .root_name = "Test",
        .estimated_total_items = test_fns.len,
    });

    var wgroup: std.Thread.WaitGroup = .{};
    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = std.heap.page_allocator });
    defer pool.deinit();

    for (test_fns) |test_fn| {
        pool.spawnWg(
            &wgroup,
            test_wrapper,
            .{ test_fn, &count, progress },
        );
    }
    wgroup.wait();
    progress.end();

    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (count.success == test_fns.len) {
        try stdout.print("All {d} tests passed.\n", .{count.success});
    } else {
        try stdout.print("{d} tests passed; {d} failed.\n", .{ count.success, count.fail });
    }
    try stdout.print("took {D}\n", .{timer.read()});

    try stdout.flush();
}

fn test_wrapper(
    test_fn: std.builtin.TestFn,
    count: *Count,
    progress: std.Progress.Node,
) void {
    const test_node = progress.start(test_fn.name, 0);
    defer test_node.end();

    if (test_fn.func()) {
        count.success += 1;
    } else |_| {
        count.fail += 1;
        if (@errorReturnTrace()) |trace| {
            std.debug.dumpStackTrace(trace.*);
        }
    }
}
