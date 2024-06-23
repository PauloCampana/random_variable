![](image.png)

For over 30 probability distributions, this library provides:

* Random variable generation
* Density, Probability and Quantile functions

# Documentation
Avaliable on <https://paulocampana.github.io/random_variable>

# Importing library
```bash
$ zig fetch --save git+https://github.com/PauloCampana/random_variable
```

```zig
// build.zig
const rv = b.dependency("random_variable", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("random_variable", rv.module("random_variable"));
```

```zig
// main.zig
const rv = @import("random_variable");
```

Or just copy the whole project into yours and import `src/root.zig`
