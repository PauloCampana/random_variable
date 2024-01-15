const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = b.addModule("random_variable", .{
        .root_source_file = .{ .path = "src/root.zig" },
    });

    const tests = b.addTest(.{
        .root_source_file = .{ .path = "src/root.zig" },
        .target = target,
        .optimize = optimize,
    });
    const test_cmd = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&test_cmd.step);

    const docs_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/root.zig" },
    });
    const docs_cmd = b.addInstallDirectory(.{
        .source_dir = docs_tests.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "../docs",
    });
    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&docs_cmd.step);
}
