#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kInvSqrt2Pi = 0.39894228040143267793994605993438;
constexpr double kInvSqrt2 = 0.70710678118654752440084436210485;
constexpr double kTiny = 1e-300;

inline double normal_pdf(double x) {
    if (std::abs(x) > 38.0) return kTiny;
    return std::max(kInvSqrt2Pi * std::exp(-0.5 * x * x), kTiny);
}
inline double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

inline double prob_abs_le(double t, double mean = 0.0, double sd = 1.0) {
    if (!(sd > 0.0) || !std::isfinite(sd)) {
        throw std::invalid_argument("sd must be finite and positive");
    }
    const double value =
        normal_cdf((t - mean) / sd) - normal_cdf((-t - mean) / sd);
    return std::clamp(value, kTiny, 1.0);
}

inline double logaddexp(double a, double b) {
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

inline double expit(double x) {
    if (x >= 0.0) {
        const double e = std::exp(-std::min(x, 700.0));
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(std::max(x, -700.0));
    return e / (1.0 + e);
}

inline double log_null_candidate(double value) {
    return std::log(normal_pdf(value));
}

inline double log_null_background(double value) {
    const double a0 = prob_abs_le(std::abs(value));
    return std::log(2.0) + std::log(normal_pdf(value)) + std::log(a0);
}

inline double signal_background_component(
    double value,
    double mean,
    double sd
) {
    const double t = std::abs(value);
    const double a0 = prob_abs_le(t);
    const double ax = prob_abs_le(t, mean, sd);
    const double fx = normal_pdf((value - mean) / sd) / sd;
    return std::max(fx * a0 + normal_pdf(value) * ax, kTiny);
}

inline double interpolate(
    const double value,
    const std::vector<double>& grid,
    const std::vector<double>& density
) {
    if (value <= grid.front()) return density.front();
    if (value >= grid.back()) return density.back();
    const auto upper = std::upper_bound(grid.begin(), grid.end(), value);
    const std::size_t right = static_cast<std::size_t>(upper - grid.begin());
    const std::size_t left = right - 1;
    const double fraction =
        (value - grid[left]) / (grid[right] - grid[left]);
    return density[left] +
        fraction * (density[right] - density[left]);
}

double trapezoid(
    const std::vector<double>& x,
    const std::vector<double>& y
) {
    double result = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        result += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) * 0.5;
    }
    return result;
}

std::vector<double> as_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& x
) {
    const auto input = x.unchecked<1>();
    std::vector<double> result(static_cast<std::size_t>(input.shape(0)));
    for (py::ssize_t i = 0; i < input.shape(0); ++i) {
        result[static_cast<std::size_t>(i)] = input(i);
    }
    return result;
}

py::array_t<double> to_array(const std::vector<double>& x) {
    py::array_t<double> result(x.size());
    auto output = result.mutable_unchecked<1>();
    for (std::size_t i = 0; i < x.size(); ++i) {
        output(static_cast<py::ssize_t>(i)) = x[i];
    }
    return result;
}

}  // namespace

py::dict density_primitives(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values
) {
    const auto x = values.unchecked<1>();
    py::array_t<double> log_f0(values.size());
    py::array_t<double> log_g0(values.size());
    py::array_t<double> p_absmax(values.size());
    auto out_f0 = log_f0.mutable_unchecked<1>();
    auto out_g0 = log_g0.mutable_unchecked<1>();
    auto out_p = p_absmax.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < x.shape(0); ++i) {
        const double value = x(i);
        const double a0 = prob_abs_le(std::abs(value));
        out_f0(i) = log_null_candidate(value);
        out_g0(i) = log_null_background(value);
        out_p(i) = std::clamp(1.0 - a0 * a0, 0.0, 1.0);
    }
    py::dict result;
    result["log_f0"] = std::move(log_f0);
    result["log_g0"] = std::move(log_g0);
    result["p_absmax"] = std::move(p_absmax);
    return result;
}

py::array_t<double> signal_component_logpdf(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
    double mean,
    double sd,
    bool background
) {
    if (!(sd > 0.0)) throw std::invalid_argument("sd must be positive");
    const auto x = values.unchecked<1>();
    py::array_t<double> result(values.size());
    auto output = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < x.shape(0); ++i) {
        const double density = background
            ? signal_background_component(x(i), mean, sd)
            : normal_pdf((x(i) - mean) / sd) / sd;
        output(i) = std::log(std::max(density, kTiny));
    }
    return result;
}

py::dict predictive_recursion_absmax(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& mean_grid,
    const py::array_t<long long, py::array::c_style | py::array::forcecast>& order,
    double decay = -0.67,
    int likelihood_table_size = 4096
) {
    const std::vector<double> z = as_vector(values);
    const std::vector<double> means = as_vector(mean_grid);
    if (z.empty() || means.size() < 2) {
        throw std::invalid_argument("values and a nontrivial mean_grid are required");
    }
    if (likelihood_table_size < 2) {
        throw std::invalid_argument("likelihood_table_size must be at least two");
    }
    const auto sweep_order = order.unchecked<1>();
    const std::size_t g = means.size();
    const double lower = means.front();
    const double upper = means.back();
    const double table_dx =
        (upper - lower) / static_cast<double>(likelihood_table_size - 1);

    std::vector<double> table_grid(
        static_cast<std::size_t>(likelihood_table_size)
    );
    std::vector<double> g0_table(
        static_cast<std::size_t>(likelihood_table_size)
    );
    std::vector<double> component_table(
        static_cast<std::size_t>(likelihood_table_size) * g
    );
    for (int row = 0; row < likelihood_table_size; ++row) {
        const double value = lower + table_dx * static_cast<double>(row);
        table_grid[static_cast<std::size_t>(row)] = value;
        g0_table[static_cast<std::size_t>(row)] =
            std::exp(log_null_background(value));
        for (std::size_t j = 0; j < g; ++j) {
            component_table[static_cast<std::size_t>(row) * g + j] =
                signal_background_component(value, means[j], 1.0);
        }
    }

    std::vector<double> theta(g, 1.0 / static_cast<double>(g));
    std::vector<double> joint(g);
    double pi0 = 1.0;
    for (py::ssize_t iteration = 0; iteration < sweep_order.shape(0);
         ++iteration) {
        const long long index = sweep_order(iteration);
        if (index < 0 || static_cast<std::size_t>(index) >= z.size()) {
            throw std::out_of_range("order contains an invalid zero-based index");
        }
        const double value = std::clamp(z[static_cast<std::size_t>(index)],
                                        lower, upper);
        const double coordinate = (value - lower) / table_dx;
        const int left = std::clamp(
            static_cast<int>(std::floor(coordinate)),
            0,
            likelihood_table_size - 2
        );
        const int right = left + 1;
        const double fraction = coordinate - static_cast<double>(left);
        const double g0 = (1.0 - fraction) *
                g0_table[static_cast<std::size_t>(left)] +
            fraction * g0_table[static_cast<std::size_t>(right)];
        for (std::size_t j = 0; j < g; ++j) {
            const double likelihood = (1.0 - fraction) *
                    component_table[static_cast<std::size_t>(left) * g + j] +
                fraction *
                    component_table[static_cast<std::size_t>(right) * g + j];
            joint[j] = theta[j] * likelihood;
        }
        const double m0 = pi0 * g0;
        const double m1 = trapezoid(means, joint);
        const double mixture = std::max(m0 + m1, kTiny);
        const double weight =
            std::pow(3.0 + static_cast<double>(iteration), decay);
        pi0 = (1.0 - weight) * pi0 + weight * m0 / mixture;
        for (std::size_t j = 0; j < g; ++j) {
            theta[j] = (1.0 - weight) * theta[j] +
                weight * joint[j] / mixture;
        }
    }

    const double alternative_mass = std::max(1.0 - pi0, 1e-12);
    std::vector<double> f1(g);
    std::vector<double> g1(g);
    for (std::size_t i = 0; i < g; ++i) {
        for (std::size_t j = 0; j < g; ++j) {
            joint[j] = theta[j] * normal_pdf(means[i] - means[j]);
        }
        f1[i] = std::max(trapezoid(means, joint) / alternative_mass, 1e-300);
        for (std::size_t j = 0; j < g; ++j) {
            joint[j] = theta[j] *
                signal_background_component(means[i], means[j], 1.0);
        }
        g1[i] = std::max(trapezoid(means, joint) / alternative_mass, 1e-300);
    }

    py::dict result;
    result["grid"] = to_array(means);
    result["theta_subdensity"] = to_array(theta);
    result["pi0"] = pi0;
    result["f1"] = to_array(f1);
    result["g1"] = to_array(g1);
    result["likelihood_table_size"] = likelihood_table_size;
    return result;
}

py::dict exact_weighted_kde(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& dataset,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& weights,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& evaluation,
    double bandwidth
) {
    const std::vector<double> data = as_vector(dataset);
    std::vector<double> weight = as_vector(weights);
    const std::vector<double> points = as_vector(evaluation);
    if (data.empty() || data.size() != weight.size()) {
        throw std::invalid_argument("dataset and weights must have equal nonzero length");
    }
    if (!(bandwidth > 0.0)) {
        throw std::invalid_argument("bandwidth must be positive");
    }
    double sum_weight = 0.0;
    for (double& value : weight) {
        value = std::max(value, 1e-15);
        sum_weight += value;
    }
    for (double& value : weight) value /= sum_weight;

    std::vector<double> f1(points.size(), 0.0);
    std::vector<double> g1(points.size(), 0.0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = 0; j < data.size(); ++j) {
            const double component =
                normal_pdf((points[i] - data[j]) / bandwidth) / bandwidth;
            f1[i] += weight[j] * component;
            g1[i] += weight[j] * signal_background_component(
                points[i], data[j], bandwidth
            );
        }
        f1[i] = std::max(f1[i], kTiny);
        g1[i] = std::max(g1[i], kTiny);
    }
    py::dict result;
    result["f1"] = to_array(f1);
    result["g1"] = to_array(g1);
    return result;
}

py::dict fast_weighted_kde_grid(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& dataset,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& weights,
    double bandwidth,
    int grid_size = 16384
) {
    const std::vector<double> data = as_vector(dataset);
    std::vector<double> weight = as_vector(weights);
    if (data.empty() || data.size() != weight.size()) {
        throw std::invalid_argument("dataset and weights must have equal nonzero length");
    }
    if (!(bandwidth > 0.0) || grid_size < 256) {
        throw std::invalid_argument("invalid bandwidth or grid_size");
    }
    // An odd grid contains zero exactly.  Both abs-max densities vanish
    // linearly at zero; omitting it creates a large relative interpolation
    // error for small background statistics.
    if (grid_size % 2 == 0) ++grid_size;
    double max_abs = 0.0;
    double sum_weight = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(data[i]));
        weight[i] = std::max(weight[i], 1e-15);
        sum_weight += weight[i];
    }
    for (double& value : weight) value /= sum_weight;

    const double bound = std::max(8.0, max_abs + 8.0 * bandwidth);
    const double dx = 2.0 * bound / static_cast<double>(grid_size - 1);
    std::vector<double> grid(static_cast<std::size_t>(grid_size));
    std::vector<double> binned(static_cast<std::size_t>(grid_size), 0.0);
    for (int i = 0; i < grid_size; ++i) {
        grid[static_cast<std::size_t>(i)] = -bound + dx * i;
    }
    for (std::size_t i = 0; i < data.size(); ++i) {
        const double coordinate =
            std::clamp((data[i] + bound) / dx, 0.0,
                       static_cast<double>(grid_size - 1));
        const int left = std::min(
            static_cast<int>(std::floor(coordinate)), grid_size - 2
        );
        const double fraction = coordinate - left;
        binned[static_cast<std::size_t>(left)] += weight[i] * (1.0 - fraction);
        binned[static_cast<std::size_t>(left + 1)] += weight[i] * fraction;
    }

    const int radius = std::max(
        1, static_cast<int>(std::ceil(8.0 * bandwidth / dx))
    );
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    for (int offset = -radius; offset <= radius; ++offset) {
        kernel[static_cast<std::size_t>(offset + radius)] =
            normal_pdf((offset * dx) / bandwidth) / bandwidth;
    }
    std::vector<double> f1(static_cast<std::size_t>(grid_size), 0.0);
    {
        py::gil_scoped_release release;
        for (int i = 0; i < grid_size; ++i) {
            double value = 0.0;
            const int lower = std::max(0, i - radius);
            const int upper = std::min(grid_size - 1, i + radius);
            for (int j = lower; j <= upper; ++j) {
                value += binned[static_cast<std::size_t>(j)] *
                    kernel[static_cast<std::size_t>(i - j + radius)];
            }
            f1[static_cast<std::size_t>(i)] = std::max(value, kTiny);
        }
    }
    double norm = trapezoid(grid, f1);
    for (double& value : f1) value /= norm;

    std::vector<double> cdf(static_cast<std::size_t>(grid_size), 0.0);
    for (int i = 1; i < grid_size; ++i) {
        cdf[static_cast<std::size_t>(i)] =
            cdf[static_cast<std::size_t>(i - 1)] +
            dx * (f1[static_cast<std::size_t>(i - 1)] +
                  f1[static_cast<std::size_t>(i)]) * 0.5;
    }
    const double cdf_norm = cdf.back();
    for (double& value : cdf) value /= cdf_norm;

    std::vector<double> g1(static_cast<std::size_t>(grid_size));
    for (int i = 0; i < grid_size; ++i) {
        const double value = grid[static_cast<std::size_t>(i)];
        const double t = std::abs(value);
        const double mass_abs = std::clamp(
            interpolate(t, grid, cdf) - interpolate(-t, grid, cdf),
            0.0, 1.0
        );
        g1[static_cast<std::size_t>(i)] = std::max(
            f1[static_cast<std::size_t>(i)] * prob_abs_le(t) +
                normal_pdf(value) * mass_abs,
            kTiny
        );
    }
    norm = trapezoid(grid, g1);
    for (double& value : g1) value /= norm;

    py::dict result;
    result["grid"] = to_array(grid);
    result["f1"] = to_array(f1);
    result["g1"] = to_array(g1);
    result["bandwidth"] = bandwidth;
    result["grid_size"] = grid_size;
    result["bound"] = bound;
    return result;
}

py::dict reweight_scores(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& background,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& candidate,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& background_signal,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& grid_array,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& f1_array,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& g1_array
) {
    const std::vector<double> b = as_vector(background);
    const std::vector<double> z = as_vector(candidate);
    const std::vector<double> q = as_vector(background_signal);
    const std::vector<double> grid = as_vector(grid_array);
    const std::vector<double> f1 = as_vector(f1_array);
    const std::vector<double> g1 = as_vector(g1_array);
    if (b.size() != z.size() || b.size() != q.size()) {
        throw std::invalid_argument("background, candidate, and signal map differ in length");
    }
    if (grid.size() < 2 || grid.size() != f1.size() ||
        grid.size() != g1.size()) {
        throw std::invalid_argument("invalid density grid");
    }

    std::vector<double> score(b.size());
    std::vector<double> log_score(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double qi = std::clamp(q[i], 1e-12, 1.0 - 1e-12);
        const double logit_background = std::log(qi) - std::log1p(-qi);
        const double log_odds =
            logit_background +
            std::log(std::max(interpolate(z[i], grid, f1), kTiny)) -
            log_null_candidate(z[i]) -
            std::log(std::max(interpolate(b[i], grid, g1), kTiny)) +
            log_null_background(b[i]);
        const double log_null_posterior = -logaddexp(0.0, log_odds);
        log_score[i] = log_null_posterior;
        score[i] = std::exp(std::max(log_null_posterior, -745.0));
    }
    py::dict result;
    result["score"] = to_array(score);
    result["log_score"] = to_array(log_score);
    return result;
}

PYBIND11_MODULE(absmax_backend, module) {
    module.doc() =
        "C++ numerical kernels for abs-max PLIS working-model adaptations";
    module.def("density_primitives", &density_primitives);
    module.def(
        "signal_component_logpdf",
        &signal_component_logpdf,
        py::arg("values"),
        py::arg("mean"),
        py::arg("sd"),
        py::arg("background")
    );
    module.def(
        "predictive_recursion_absmax",
        &predictive_recursion_absmax,
        py::arg("values"),
        py::arg("mean_grid"),
        py::arg("order"),
        py::arg("decay") = -0.67,
        py::arg("likelihood_table_size") = 4096
    );
    module.def(
        "exact_weighted_kde",
        &exact_weighted_kde,
        py::arg("dataset"),
        py::arg("weights"),
        py::arg("evaluation"),
        py::arg("bandwidth")
    );
    module.def(
        "fast_weighted_kde_grid",
        &fast_weighted_kde_grid,
        py::arg("dataset"),
        py::arg("weights"),
        py::arg("bandwidth"),
        py::arg("grid_size") = 16384
    );
    module.def(
        "reweight_scores",
        &reweight_scores,
        py::arg("background"),
        py::arg("candidate"),
        py::arg("background_signal"),
        py::arg("grid"),
        py::arg("f1"),
        py::arg("g1")
    );
}
