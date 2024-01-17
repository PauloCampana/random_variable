![](image.png)

For over 20 probability distributions, this library provides:

* Random variable generation
* Density, Distribution and Quantile functions

# Documentation
Avaliable on <https://paulocampana.github.io/random_variable>

# Importing library
Requires Zig version 0.12.0-dev.2133 or above.

Grab the commit hash of the version you want to use and in your `build.zig.zon`, add inside the dependencies:

```zig
.random_variable = .{
    .url = "https://github.com/paulocampana/random_variable/archive/{commit_hash_here}.tar.gz",
    // .hash = "leave this commented, compiler will tell you what to put here",
},
```

You could also copy the whole repository somewhere and use the `.path` field instead of `.url`

Then, on your `build.zig`, add the `dependency` at the top and the `addImport` for every exe/lib/tests you need:

```zig
// ...
const random_variable = b.dependency("random_variable", .{
    .target = target,
    .optimize = optimize,
});
// ...
exe.root_module.addImport("random_variable", random_variable.module("random_variable"));
```

You can then import the library where you need

```zig
const rv = @import("random_variable");
```
