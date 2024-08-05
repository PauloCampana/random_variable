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

    if (count.success == test_fns.len) {
        std.debug.print("All {d} tests passed.\n", .{count.success});
    } else {
        std.debug.print("{d} tests passed; {d} failed.\n", .{ count.success, count.fail });
    }
    std.debug.print("took {}\n", .{std.fmt.fmtDuration(timer.read())});
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
