const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const module = b.addModule("random_variable", .{
        .root_source_file = b.path("src/root.zig"),
    });

    const lib = b.addStaticLibrary(.{
        .name = "random_variable",
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    const header = b.addInstallHeaderFile(b.path("src/random_variable.h"), "random_variable.h");
    b.getInstallStep().dependOn(&header.step);

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

    const top_level_docs = b.addInstallDirectory(.{
        .source_dir = tests.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "../docs",
    });
    const docs = b.addInstallDirectory(.{
        .source_dir = tests.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&top_level_docs.step);
    docs_step.dependOn(&docs.step);
}
