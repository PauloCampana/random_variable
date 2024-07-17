![](image.png)

For over 30 probability distributions, this library provides:

* Random variable generation
* Density, probability, survival and quantile functions

Mathematical functions are tested for high precision and
RNG tested to correctly fit the distribution

# Installation
Builds static library with header file, optionally run tests
```bash
$ zig build -Doptimize=ReleaseFast
$ zig build test
$ zig build correctness -Doptimize=ReleaseFast
```

# [Documentation](https://paulocampana.github.io/random_variable)

# Importing Zig module
```bash
$ zig fetch --save git+https://github.com/PauloCampana/random_variable
```

```zig
// build.zig
const rv_dep = b.dependency("random_variable", .{
    .target = target,
    .optimize = optimize,
});
const rv_mod = rv_dep.module("random_variable");

exe.root_module.addImport("random_variable", rv_mod);
```

```zig
// main.zig
const rv = @import("random_variable");
```
