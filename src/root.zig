//! Density, probability, survival and quantile functions +
//! random number generation for probability distributions.
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe modes
//! such as ±NaN, ±Inf, probabilities outside of the [0,1] interval or
//! negative shape/scale parameters
//!
//! Random variable generation has 2 flavours: `random` and `fill`,
//!   * `random` returns a single random variable
//!   * `fill` fills a buffer with random variables
//!
//! it might be faster to call `fill` once than `random` in a loop due to
//! using a different algorithm that is faster but has a setup time.

// zig fmt: off
pub const benford              = @import("distribution/benford.zig");
pub const bernoulli            = @import("distribution/bernoulli.zig");
pub const beta                 = @import("distribution/beta.zig");
pub const beta_binomial        = @import("distribution/beta_binomial.zig");
pub const beta_prime           = @import("distribution/beta_prime.zig");
pub const binomial             = @import("distribution/binomial.zig");
pub const cauchy               = @import("distribution/cauchy.zig");
pub const chi                  = @import("distribution/chi.zig");
pub const chi_squared          = @import("distribution/chi_squared.zig");
pub const continuous_bernoulli = @import("distribution/continuous_bernoulli.zig");
pub const dagum                = @import("distribution/dagum.zig");
pub const discrete_uniform     = @import("distribution/discrete_uniform.zig");
pub const exponential          = @import("distribution/exponential.zig");
pub const f                    = @import("distribution/f.zig");
pub const gamma                = @import("distribution/gamma.zig");
pub const geometric            = @import("distribution/geometric.zig");
pub const gompertz             = @import("distribution/gompertz.zig");
pub const gumbel               = @import("distribution/gumbel.zig");
pub const hypergeometric       = @import("distribution/hypergeometric.zig");
pub const laplace              = @import("distribution/laplace.zig");
pub const logarithmic          = @import("distribution/logarithmic.zig");
pub const logistic             = @import("distribution/logistic.zig");
pub const log_normal           = @import("distribution/log_normal.zig");
pub const negative_binomial    = @import("distribution/negative_binomial.zig");
pub const normal               = @import("distribution/normal.zig");
pub const pareto               = @import("distribution/pareto.zig");
pub const poisson              = @import("distribution/poisson.zig");
pub const rayleigh             = @import("distribution/rayleigh.zig");
pub const t                    = @import("distribution/t.zig");
pub const uniform              = @import("distribution/uniform.zig");
pub const weibull              = @import("distribution/weibull.zig");
// zig fmt: on

comptime {
    refAllDeclsRecursive(@This());
}

// std.testing.refAllDeclsRecursive but works outside tests,
// needed to analyze all the namespaces and export functions
fn refAllDeclsRecursive(comptime T: type) void {
    inline for (@typeInfo(T).@"struct".decls) |decl| {
        const field = @field(T, decl.name);
        if (@TypeOf(field) == type) {
            switch (@typeInfo(field)) {
                .@"struct",
                .@"enum",
                .@"union",
                .@"opaque",
                => refAllDeclsRecursive(field),
                else => {},
            }
        }
        _ = &field;
    }
}
