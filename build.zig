const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const module = b.addModule("random_variable", .{
        .root_source_file = b.path("src/root.zig"),
    });

    const tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const test_run = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&test_run.step);

    const gof = b.addTest(.{
        .name = "gof",
        .root_source_file = b.path("src/gof_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    gof.root_module.addImport("random_variable", module);
    const gof_run = b.addRunArtifact(gof);
    const gof_step = b.step("gof", "Run \"Goodness of fit\" tests for random variable generation");
    gof_step.dependOn(&gof_run.step);

    const docs = b.addObject(.{
        .name = "docs",
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("src/root.zig"),
    });
    const docs_run = b.addInstallDirectory(.{
        .source_dir = docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "../docs",
    });
    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&docs_run.step);
}
