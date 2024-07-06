//! Density/mass, probability and quantile functions for probability distributions.
//!
//! Asserts invalid distribution parameters on Debug and ReleaseSafe modes
//! such as ±NaN, ±Inf, probabilities outside of the [0,1] interval or
//! negative shape/scale parameters

#ifndef RANDOM_VARIABLE_H
#define RANDOM_VARIABLE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

//! Support: {1,2,⋯,b - 1}
//!
//! Parameters:
//! - b: `base` ∈ {2,3,4,⋯}

/// p(x) = log_b(1 + 1 / x)
double rv_benford_density(double x, uint64_t base);
/// F(q) = log_b(1 + ⌊q⌋)
double rv_benford_probability(double q, uint64_t base);
/// S(t) = log_b(b / (1 + ⌊t⌋))
double rv_benford_survival(double t, uint64_t base);
/// Q(p) = ⌈b^p⌉ - 1
double rv_benford_quantile(double p, uint64_t base);

//! Support: {0,1}
//!
//! Parameters:
//! - p: `prob` ∈ [0,1]

/// p(x) = 1 - p, x = 0
///
/// p(x) = p    , x = 1
double rv_bernoulli_density(double x, double prob);
/// F(q) = 0    ,      q < 0
///
/// F(q) = 1 - p, 0 <= q < 1
///
/// F(q) = 1    , 1 <= q
double rv_bernoulli_probability(double q, double prob);
/// S(t) = 1,      t < 0
///
/// S(t) = p, 0 <= t < 1
///
/// S(t) = 0, 1 <= t
double rv_bernoulli_survival(double t, double prob);
/// Q(x) = 0, x <= 1 - p
///
/// Q(x) = 1, x >  1 - p
double rv_bernoulli_quantile(double p, double prob);

//! Support: [0,1]
//!
//! Parameters:
//! - α: `shape1` ∈ (0,∞)
//! - β: `shape2` ∈ (0,∞)

/// f(x) = x^(α - 1) (1 - x)^(β - 1) / beta(α, β)
double rv_beta_density(double x, double shape1, double shape2);
/// No closed form
double rv_beta_probability(double q, double shape1, double shape2);
/// No closed form
double rv_beta_survival(double t, double shape1, double shape2);
/// No closed form
double rv_beta_quantile(double p, double shape1, double shape2);

//! Support: {0,1,⋯,n}
//!
//! Parameters:
//! - n: `size`   ∈ {0,1,2,⋯}
//! - α: `shape1` ∈ (0,∞)
//! - β: `shape2` ∈ (0,∞)

/// p(x) = (n x) beta(x + α, n - x + β) / beta(α, β)
double rv_beta_binomial_density(double x, uint64_t size, double shape1, double shape2);
/// No closed form
double rv_beta_binomial_probability(double q, uint64_t size, double shape1, double shape2);
/// No closed form
double rv_beta_binomial_survival(double t, uint64_t size, double shape1, double shape2);
/// No closed form
double rv_beta_binomial_quantile(double p, uint64_t size, double shape1, double shape2);

//! Support: [0,∞)
//!
//! Parameters:
//! - α: `shape1` ∈ (0,∞)
//! - β: `shape2` ∈ (0,∞)

/// f(x) = x^(α - 1) (1 + x)^(-α - β) / beta(α, β)
double rv_beta_prime_density(double x, double shape1, double shape2);
/// No closed form
double rv_beta_prime_probability(double q, double shape1, double shape2);
/// No closed form
double rv_beta_prime_survival(double t, double shape1, double shape2);
/// No closed form
double rv_beta_prime_quantile(double p, double shape1, double shape2);

//! Support: {0,1,⋯,n}
//!
//! Parameters:
//! - n: `size` ∈ {0,1,2,⋯}
//! - p: `prob` ∈ [0,1]

/// p(x) = (n x) p^x (1 - p)^(n - x)
double rv_binomial_density(double x, uint64_t size, double prob);
/// No closed form
double rv_binomial_probability(double q, uint64_t size, double prob);
/// No closed form
double rv_binomial_survival(double t, uint64_t size, double prob);
/// No closed form
double rv_binomial_quantile(double p, uint64_t size, double prob);

//! Support: (-∞,∞)
//!
//! Parameters:
//! - μ: `location` ∈ (-∞,∞)
//! - σ: `scale`    ∈ ( 0,∞)

/// f(x) = 1 / (πσ (1 + ((x - μ) / σ)^2))
double rv_cauchy_density(double x, double location, double scale);
/// F(q) = 0.5 + arctan((q - μ) / σ) / π
double rv_cauchy_probability(double q, double location, double scale);
/// S(t) = 0.5 - arctan((t - μ) / σ) / π
double rv_cauchy_survival(double q, double location, double scale);
/// Q(p) = μ + σ tan(π (p - 0.5))
double rv_cauchy_quantile(double p, double location, double scale);

//! Support: [0,∞)
//!
//! Parameters:
//! - ν: `df` ∈ (0,∞)

/// f(x) = x^(ν - 1) exp(-x^2 / 2) / (2^(ν / 2 - 1) gamma(ν / 2))
double rv_chi_density(double x, double df);
/// No closed form
double rv_chi_probability(double q, double df);
/// No closed form
double rv_chi_survival(double t, double df);
/// No closed form
double rv_chi_quantile(double p, double df);

//! Support: [0,∞)
//!
//! Parameters:
//! - ν: `df` ∈ (0,∞)

/// f(x) = 0.5 / gamma(ν / 2) (x / 2)^(ν / 2 - 1) exp(-x / 2)
double rv_chi_squared_density(double x, double df);
/// No closed form
double rv_chi_squared_probability(double q, double df);
/// No closed form
double rv_chi_squared_survival(double t, double df);
/// No closed form
double rv_chi_squared_quantile(double p, double df);

//! Support: [0,1]
//!
//! Parameters:
//! - λ: `shape` ∈ (0,1)

/// f(x) = 2 / (1 - 2λ) arctanh(1 - 2λ) λ^x (1 - λ)^(1 - x)
double rv_continuous_bernoulli_density(double x, double shape);
/// F(q) = (λ^q (1 - λ)^(1 - q) + λ - 1) / (2λ - 1)
double rv_continuous_bernoulli_probability(double q, double shape);
/// S(t) = (λ - λ^t (1 - λ)^(1 - t)) / (2λ - 1)
double rv_continuous_bernoulli_survival(double t, double shape);
/// Q(p) = ln(((2λ - 1)p - λ + 1) / (1 - λ)) / ln(λ / (1 - λ))
double rv_continuous_bernoulli_quantile(double p, double shape);

//! Support: [0,∞)
//!
//! Parameters:
//! - p: `shape1` ∈ (0,∞)
//! - α: `shape2` ∈ (0,∞)
//! - σ: `scale`  ∈ (0,∞)

/// f(x) = pα / σ (x / σ)^(pα - 1) / (1 + (x / σ)^α)^(p + 1)
double rv_dagum_density(double x, double shape1, double shape2, double scale);
/// F(q) = (1 + (q / σ)^-α)^-p
double rv_dagum_probability(double q, double shape1, double shape2, double scale);
/// S(t) = 1 - (1 + (t / σ)^-α)^-p
double rv_dagum_survival(double t, double shape1, double shape2, double scale);
/// Q(x) = σ(x^(-1 / p) - 1)^(- 1 / α)
double rv_dagum_quantile(double p, double shape1, double shape2, double scale);

//! Support: {a,⋯,b}
//!
//! Parameters:
//! - a: `min` ∈ {⋯,-1,0,1,⋯}
//! - b: `max` ∈ {a,a + 1,⋯}

/// p(x) = 1 / (b - a + 1)
double rv_discrete_uniform_density(double x, int64_t min, int64_t max);
/// F(q) = (⌊q⌋ - a + 1) / (b - a + 1)
double rv_discrete_uniform_probability(double q, int64_t min, int64_t max);
/// S(t) = (b - ⌊t⌋) / (b - a + 1)
double rv_discrete_uniform_survival(double t, int64_t min, int64_t max);
/// Q(p) = ⌈p (b - a + 1)⌉ + a - 1
double rv_discrete_uniform_quantile(double p, int64_t min, int64_t max);

//! Support: [0,∞)
//!
//! Parameters:
//! - σ: `scale` ∈ (0,∞)

/// f(x) = exp(-x / σ) / σ
double rv_exponential_density(double x, double scale);
/// F(q) = 1 - exp(-q / σ)
double rv_exponential_probability(double q, double scale);
/// S(t) = exp(-t / σ)
double rv_exponential_survival(double t, double scale);
/// Q(p) = -σ ln(1 - p)
double rv_exponential_quantile(double p, double scale);

//! Support: [0,∞)
//!
//! Parameters:
//! - n: `df1` ∈ (0,∞)
//! - m: `df2` ∈ (0,∞)

/// f(x) = n^(n / 2) m^(m / 2) x^(n / 2 - 1) (m + nx)^(-(n + m) / 2) / beta(n / 2, m / 2)
double rv_f_density(double x, double df1, double df2);
/// No closed form
double rv_f_probability(double q, double df1, double df2);
/// No closed form
double rv_f_survival(double t, double df1, double df2);
/// No closed form
double rv_f_quantile(double p, double df1, double df2);

//! Support: [0,∞)
//!
//! Parameters:
//! - α: `shape` ∈ (0,∞)
//! - σ: `scale` ∈ (0,∞)

/// f(x) = 1 / (σ gamma(α)) (x / σ)^(α - 1) exp(-x / σ)
double rv_gamma_density(double x, double shape, double scale);
/// No closed form
double rv_gamma_probability(double q, double shape, double scale);
/// No closed form
double rv_gamma_survival(double t, double shape, double scale);
/// No closed form
double rv_gamma_quantile(double p, double shape, double scale);

//! Support: {0,1,2,∞}
//!
//! Parameters:
//! - p: `prob` ∈ (0,1]

/// p(x) = p (1 - p)^x
double rv_geometric_density(double x, double prob);
/// F(q) = 1 - (1 - p)^(⌊q⌋ + 1)
double rv_geometric_probability(double q, double prob);
/// S(t) = (1 - p)^(⌊t⌋ + 1)
double rv_geometric_survival(double t, double prob);
/// Q(x) = ⌊ln(1 - x) / ln(1 - p)⌋
double rv_geometric_quantile(double p, double prob);

//! Support: [0,∞)
//!
//! Parameters:
//! - α: `shape` ∈ (0,∞)
//! - σ: `scale` ∈ (0,∞)

/// f(x) = α / σ exp(α(1 - exp(x / σ)) + x / σ)
double rv_gompertz_density(double x, double shape, double scale);
/// F(q) = 1 - exp(α(1 - exp(q / σ)))
double rv_gompertz_probability(double q, double shape, double scale);
/// S(t) = exp(α(1 - exp(t / σ)))
double rv_gompertz_survival(double t, double shape, double scale);
/// Q(p) = σ ln(1 - ln(1 - p) / α)
double rv_gompertz_quantile(double p, double shape, double scale);

//! Support: (-∞,∞)
//!
//! Parameters:
//! - μ: `location` ∈ (-∞,∞)
//! - σ: `scale`    ∈ ( 0,∞)

/// f(x) = exp(-(x - μ) / σ - exp(-(x - μ) / σ)) / σ
double rv_gumbel_density(double x, double location, double scale);
/// F(q) = exp(-exp(-(q - μ) / σ))
double rv_gumbel_probability(double q, double location, double scale);
/// S(t) = 1 - exp(-exp(-(t - μ) / σ))
double rv_gumbel_survival(double t, double location, double scale);
/// Q(p) = μ - σ ln(-ln(p))
double rv_gumbel_quantile(double p, double location, double scale);

//! Support: {max(0, n + K - N),1,⋯,min(n, K)}
//!
//! Parameters:
//! - N: `total` ∈ {0,1,2,⋯}
//! - K: `good`  ∈ {0,1,⋯,N}
//! - n: `tries` ∈ {0,1,⋯,N}

/// p(x) = (K x) (N - K n - x) / (N n)
double rv_hypergeometric_density(double x, uint64_t total, uint64_t good, uint64_t tries);
/// No closed form
double rv_hypergeometric_probability(double q, uint64_t total, uint64_t good, uint64_t tries);
/// No closed form
double rv_hypergeometric_survival(double t, uint64_t total, uint64_t good, uint64_t tries);
/// No closed form
double rv_hypergeometric_quantile(double p, uint64_t total, uint64_t good, uint64_t tries);

//! Support: (-∞,∞)
//!
//! Parameters:
//! - μ: `location` ∈ (-∞,∞)
//! - σ: `scale`    ∈ ( 0,∞)

/// f(x) = exp(-|x - μ| / σ) / 2σ
double rv_laplace_density(double x, double location, double scale);
/// F(q) =     exp(+(q - μ) / σ)) / 2, x < μ
///
/// F(q) = 1 - exp(-(q - μ) / σ)) / 2, x > μ
double rv_laplace_probability(double q, double location, double scale);
/// S(t) = 1 - exp(+(t - μ) / σ)) / 2, x < μ
///
/// S(t) =     exp(-(t - μ) / σ)) / 2, x > μ
double rv_laplace_survival(double t, double location, double scale);
/// Q(p) = μ + σ ln(2p)      , 0.0 < p < 0.5
///
/// Q(p) = μ - σ ln(2(1 - p)), 0.5 < p < 1.0
double rv_laplace_quantile(double p, double location, double scale);

//! Support: {1,2,3,⋯}
//!
//! Parameters:
//! - p: `prob` ∈ (0,1)

/// p(x) = p^x / (-ln(1 - p) x)
double rv_logarithmic_density(double x, double prob);
/// No closed form
double rv_logarithmic_probability(double q, double prob);
/// No closed form
double rv_logarithmic_survival(double t, double prob);
/// No closed form
double rv_logarithmic_quantile(double p, double prob);

//! Support: (-∞,∞)
//!
//! Parameters:
//! - μ: `location` ∈ (-∞,∞)
//! - σ: `scale`    ∈ ( 0,∞)

/// f(x) = exp(-(x - μ) / σ) / (σ (1 + exp(-(x - μ) / σ))^2)
double rv_logistic_density(double x, double location, double scale);
/// F(q) = 1 / (1 + exp(-(q - μ) / σ))
double rv_logistic_probability(double q, double location, double scale);
/// S(t) = 1 / (1 + exp((t - μ) / σ))
double rv_logistic_survival(double t, double location, double scale);
/// Q(p) = μ + σ ln(p / (1 - p))
double rv_logistic_quantile(double p, double location, double scale);

//! Support: [0,∞)
//!
//! Parameters:
//! - μ: `log_location` ∈ (-∞,∞)
//! - σ: `log_scale`    ∈ ( 0,∞)

/// f(x) = exp(-((ln(x) - μ) / σ)^2 / 2) / (xσ sqrt(2π))
double rv_log_normal_density(double x, double log_location, double log_scale);
/// No closed form
double rv_log_normal_probability(double q, double log_location, double log_scale);
/// No closed form
double rv_log_normal_survival(double t, double log_location, double log_scale);
/// No closed form
double rv_log_normal_quantile(double p, double log_location, double log_scale);

//! Support: {0,1,2,⋯}
//!
//! Parameters:
//! - n: `size` ∈ {1,2,⋯}
//! - p: `prob` ∈ (0,1]

/// p(x) = (x + n - 1 x) p^n (1 - p)^x
double rv_negative_binomial_density(double x, uint64_t size, double prob);
/// No closed form
double rv_negative_binomial_probability(double q, uint64_t size, double prob);
/// No closed form
double rv_negative_binomial_survival(double t, uint64_t size, double prob);
/// No closed form
double rv_negative_binomial_quantile(double p, uint64_t size, double prob);

//! Support: (-∞,∞)
//!
//! Parameters:
//! - μ: `location` ∈ (-∞,∞)
//! - σ: `scale`    ∈ ( 0,∞)

/// f(x) = exp(-((x - μ) / σ)^2 / 2) / (σ sqrt(2π))
double rv_normal_density(double x, double location, double scale);
/// No closed form
double rv_normal_probability(double q, double location, double scale);
/// No closed form
double rv_normal_survival(double t, double location, double scale);
/// No closed form
double rv_normal_quantile(double p, double location, double scale);

//! Support: [k,∞)
//!
//! Parameters:
//! - α: `shape`   ∈ (0,∞)
//! - k: `minimum` ∈ (0,∞)

/// f(x) = αk^α / x^(α + 1)
double rv_pareto_density(double x, double shape, double minimum);
/// F(q) = 1 - (k / q)^α
double rv_pareto_probability(double q, double shape, double minimum);
/// S(t) = (k / t)^α
double rv_pareto_survival(double t, double shape, double minimum);
/// Q(p) = k / (1 - p)^(1 / α)
double rv_pareto_quantile(double p, double shape, double minimum);

//! Support: {0,1,2,⋯}
//!
//! Parameters:
//! - λ: `lambda` ∈ (0,∞)

/// p(x) = λ^x exp(-λ) / x!
double rv_poisson_density(double x, double lambda);
/// No closed form
double rv_poisson_probability(double q, double lambda);
/// No closed form
double rv_poisson_survival(double t, double lambda);
/// No closed form
double rv_poisson_quantile(double p, double lambda);

//! Support: [0,∞)
//!
//! Parameters:
//! - σ: `scale` ∈ (0,∞)

/// f(x) = x / σ^2 exp(-x^2 / 2σ^2))
double rv_rayleigh_density(double x, double scale);
/// F(q) = 1 - exp(-q^2 / 2σ^2)
double rv_rayleigh_probability(double q, double scale);
/// S(t) = exp(-t^2 / 2σ^2)
double rv_rayleigh_survival(double t, double scale);
/// Q(p) = σ sqrt(-2ln(1 - p))
double rv_rayleigh_quantile(double p, double scale);

//! Support: (-∞,∞)
//!
//! Parameters:
//! - ν: `df` ∈ (0,∞)

/// f(x) (ν / (ν + x^2))^((ν + 1) / 2) / (sqrt(ν) beta(ν / 2, 1 / 2))
double rv_t_density(double x, double df);
/// No closed form
double rv_t_probability(double q, double df);
/// No closed form
double rv_t_survival(double t, double df);
/// No closed form
double rv_t_quantile(double p, double df);

//! Support: [a,b]
//!
//! Parameters:
//! - a: `min` ∈ (-∞,∞)
//! - b: `max` ∈ [ a,∞)

/// f(x) = 1 / (b - a)
double rv_uniform_density(double x, double min, double max);
/// F(q) = (q - a) / (b - a)
double rv_uniform_probability(double q, double min, double max);
/// S(t) = (b - t) / (b - a)
double rv_uniform_survival(double t, double min, double max);
/// Q(p) = a + (b - a)p
double rv_uniform_quantile(double p, double min, double max);

//! Support: [0,∞)
//!
//! Parameters:
//! - α: `shape` ∈ (0,∞)
//! - σ: `scale` ∈ (0,∞)

/// f(x) = α / σ (x / σ)^(α - 1) exp(-(x / σ)^α)
double rv_weibull_density(double x, double shape, double scale);
/// F(q) = 1 - exp(-(q / σ)^α)
double rv_weibull_probability(double q, double shape, double scale);
/// S(t) = exp(-(t / σ)^α)
double rv_weibull_survival(double t, double shape, double scale);
/// Q(p) = σ (-ln(1 - p))^(1 / α)
double rv_weibull_quantile(double p, double shape, double scale);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // RANDOM_VARIABLE_H
