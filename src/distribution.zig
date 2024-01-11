//! Density/mass, probability, quantile and random number generation
//! functions for common probability distributions
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe modes
//! such as ±NaN, ±Inf, probabilities outside of the [0,1] interval and
//! certain parameters with value zero or negative.
//!
//! Random variable generation has 3 flavours: `single`, `buffer` and `alloc`,
//! `single` returns one generated number, `buffer` fills a slice with generated numbers,
//! `alloc` takes an allocator and the quantity to be generated then returns a heap allocated
//! slice, result must be freed by the caller.

// discrete
pub const bernoulli = @import("distribution/bernoulli.zig");
pub const geometric = @import("distribution/geometric.zig");
pub const binomial = @import("distribution/binomial.zig");
pub const negativeBinomial = @import("distribution/negativeBinomial.zig");
pub const poisson = @import("distribution/poisson.zig");
pub const hypergeometric = @import("distribution/hypergeometric.zig");
pub const benford = @import("distribution/benford.zig");
pub const betabinomial = @import("distribution/betabinomial.zig");

// continuous
pub const uniform = @import("distribution/uniform.zig");
pub const exponential = @import("distribution/exponential.zig");
pub const weibull = @import("distribution/weibull.zig");
pub const cauchy = @import("distribution/cauchy.zig");
pub const logistic = @import("distribution/logistic.zig");
pub const gamma = @import("distribution/gamma.zig");
pub const chiSquared = @import("distribution/chiSquared.zig");
pub const f = @import("distribution/f.zig");
pub const beta = @import("distribution/beta.zig");
pub const normal = @import("distribution/normal.zig");
pub const logNormal = @import("distribution/logNormal.zig");
pub const t = @import("distribution/t.zig");
pub const betaprime = @import("distribution/betaprime.zig");

// Categorical
// Negative hypergeometric
// Fisher's noncentral hypergeometric
// Wallenius' noncentral hypergeometric
// Poisson binomial
// Rademacher
// Soliton (ideal and robust)
// Discrete uniform
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
// Logarithmic (series)
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
// Continuous Bernoulli

// Henyey-Greenstein phase function
// Mie phase function
// Von Mises
// Wrapped normal
// Wrapped exponential
// Wrapped Lévy
// Wrapped Cauchy
// Wrapped Laplace
// Wrapped asymmetric Laplace
// Dirac comb

// Birnbaum-Saunders
// Burr
// Chi
// Noncentral chi
// Inverse-chi-squared
// Noncentral chi-squared
// Scaled inverse chi-squared
// Dagum
// Exponential-logarithmic
// Noncentral F
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
// Pareto
// Pearson Type III
// Phase-type
// Phased bi-exponential
// Phased bi-Weibull
// Semi-bounded quantile-parameterized distributions
// Rayleigh
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
// Laplace
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
