//! Density/mass, probability, quantile and random number generation
//! functions for probability distributions.
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe modes
//! such as ±NaN, ±Inf, probabilities outside of the [0,1] interval and
//! certain parameters with value zero or negative.
//!
//! Random variable generation has 2 flavours: `single` and `fill`,
//! `single` returns one random variable,
//! `fill` fills a buffer with random variables, it might be faster
//! than calling `single` in a loop due to using a different algorithm
//! that is faster but has setup time.

// zig fmt: off
pub const benford             = @import("distribution/benford.zig");
pub const bernoulli           = @import("distribution/bernoulli.zig");
pub const beta                = @import("distribution/beta.zig");
pub const betaBinomial        = @import("distribution/betaBinomial.zig");
pub const betaPrime           = @import("distribution/betaPrime.zig");
pub const binomial            = @import("distribution/binomial.zig");
pub const cauchy              = @import("distribution/cauchy.zig");
pub const chi                 = @import("distribution/chi.zig");
pub const chiSquared          = @import("distribution/chiSquared.zig");
pub const continuousBernoulli = @import("distribution/continuousBernoulli.zig");
pub const dagum               = @import("distribution/dagum.zig");
pub const discreteUniform     = @import("distribution/discreteUniform.zig");
pub const exponential         = @import("distribution/exponential.zig");
pub const f                   = @import("distribution/f.zig");
pub const gamma               = @import("distribution/gamma.zig");
pub const geometric           = @import("distribution/geometric.zig");
pub const hypergeometric      = @import("distribution/hypergeometric.zig");
pub const laplace             = @import("distribution/laplace.zig");
pub const logarithmic         = @import("distribution/logarithmic.zig");
pub const logistic            = @import("distribution/logistic.zig");
pub const logNormal           = @import("distribution/logNormal.zig");
pub const negativeBinomial    = @import("distribution/negativeBinomial.zig");
pub const normal              = @import("distribution/normal.zig");
pub const pareto              = @import("distribution/pareto.zig");
pub const poisson             = @import("distribution/poisson.zig");
pub const rayleigh            = @import("distribution/rayleigh.zig");
pub const t                   = @import("distribution/t.zig");
pub const uniform             = @import("distribution/uniform.zig");
pub const weibull             = @import("distribution/weibull.zig");
// zig fmt: on

test {
    @import("std").testing.refAllDeclsRecursive(@This());
}

// Categorical
// Negative hypergeometric
// Fisher's noncentral hypergeometric
// Wallenius' noncentral hypergeometric
// Poisson binomial
// Rademacher
// Soliton (ideal and robust)
// Zipf's
// Zipf-Mandelbrot

// Beta negative binomial
// Boltzmann
// Gibbs
// Borel
// Conway-Maxwell-Poisson
// Delaporte
// Discrete compound Poisson
// Discrete phase-type
// Displaced Poisson
// Extended negative binomial
// Flory-Schulz
// Gauss-Kuzmin
// General Poisson binomial
// Generalized log-series
// Hardy
// Hermite
// Hyper-Poisson
// Mixed Poisson
// Panjer
// Polya-Eggenberger
// Parabolic fractal
// Poisson type
// Yule-Simon
// Skellam
// Skew elliptical
// Zero-truncated Poisson
// Zeta

// Degenerate

// Maxwell-Boltzmann
// Four-parameter Beta
// Generalized beta prime
// Arcsine
// PERT
// Irwin–Hall
// Bates
// Logit-normal
// Dirac delta
// Ken
// Kumaraswamy
// Logit metalog
// Marchenko-Pastur
// Bounded quantile-parameterized
// Raised cosine
// Reciprocal
// Triangular
// Trapezoidal
// Truncated normal
// U-quadratic
// Von Mises-Fisher
// Bingham
// Wigner semicircle

// Birnbaum-Saunders
// Burr
// Inverse-chi-squared
// Scaled inverse chi-squared
// Exponential-logarithmic
// Folded normal
// Fréchet
// Erlang
// Inverse-gamma
// Generalized gamma
// Generalized Pareto
// Gamma/Gompertz
// Gompertz
// Half-normal
// Hotelling's T-squared
// Inverse Gaussian (Wald)
// Lévy
// Lindley
// Log-Cauchy
// Log-Laplace
// Log-logistic
// Log-metalog
// Lomax
// Mittag-Leffler
// Nakagami
// Pearson Type III
// Phase-type
// Phased bi-exponential
// Phased bi-Weibull
// Semi-bounded quantile-parameterized distributions
// Rayleigh mixture
// Rice
// Shifted Gompertz
// Type-2 Gumbel
// Modified half-normal
// Polya-Gamma
// Modified Polya-gamma

// Behrens-Fisher
// Centralized inverse-Fano
// Chernoff's
// Exponentially modified Gaussian
// Expectile
// Fisher-Tippett, (extreme value, log-Weibull)
// Fisher's z
// skewed generalized t
// Gamma-difference
// Generalized logistic
// Generalized normal
// Geometric stable
// Gumbel
// Holtsmark
// Hyperbolic
// Hyperbolic secant
// Johnson SU
// Landau
// Lévy skew alpha-stable
// Linnik
// Map-Airy
// Metalog
// Normal-exponential-gamma
// Normal-inverse Gaussian
// Pearson Type IV
// Pearson distributions
// Quantile-parameterized
// Skew normal distribution
// Noncentral t
// Skew t distribution
// Skew laplace
// Champernowne
// Type-1 Gumbel
// Tracy-Widom
// Voigt
// Chen

// Generalized extreme value
// Generalized Pareto
// Metalog
// Tukey lambda
// Wakeby

// Rectified Gaussian
// Compound poisson-gamma (Tweedie)

// Cantor
// generalized logistic family
// metalog family
// Pearson family
// phase-type
