const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "Omit debug symbols");

    const module = b.addModule("random_variable", .{
        .root_source_file = b.path("src/root.zig"),
    });

    const lib = b.addStaticLibrary(.{
        .name = "random_variable",
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
    });
    b.installArtifact(lib);

    const header = b.addInstallHeaderFile(b.path("src/random_variable.h"), "random_variable.h");
    b.getInstallStep().dependOn(&header.step);

    const tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
    });
    const test_run = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&test_run.step);

    const correctness = b.addExecutable(.{
        .name = "correctness",
        .root_source_file = b.path("src/correctness.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
    });
    correctness.root_module.addImport("random_variable", module);
    const correctness_run = b.addRunArtifact(correctness);
    const correctness_step = b.step("correctness", "Run Kolmogorov \"goodness of fit\" test on random functions");
    correctness_step.dependOn(&correctness_run.step);

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
