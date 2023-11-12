const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "RandomVariable",
        .root_source_file = .{.path = "src/main.zig"},
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    const tests = b.addTest(.{
        .root_source_file = .{.path = "src/main.zig"},
        .target = target,
        .optimize = optimize,
    });
    const test_cmd = b.addRunArtifact(tests);
    test_cmd.step.dependOn(b.getInstallStep());
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&test_cmd.step);
}
