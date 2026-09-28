#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
#include "graph_fl.h"
#include "tf.h"
}

namespace py = pybind11;

class PersistentGFLWorkspace {
public:
    PersistentGFLWorkspace(
        int nnodes,
        const py::array_t<int, py::array::c_style | py::array::forcecast>& trails,
        const py::array_t<int, py::array::c_style | py::array::forcecast>& breakpoints,
        int num_threads = 1
    ) : n_(nnodes), num_threads_(num_threads) {
        if (n_ <= 0 || trails.size() <= 0 || breakpoints.size() <= 0) {
            throw std::invalid_argument("nonempty graph data are required");
        }
        if (num_threads_ <= 0) {
            throw std::invalid_argument("num_threads must be positive");
        }
        const auto trail_input = trails.unchecked<1>();
        const auto break_input = breakpoints.unchecked<1>();
        trails_.resize(static_cast<std::size_t>(trail_input.shape(0)));
        breakpoints_.resize(static_cast<std::size_t>(break_input.shape(0)));
        for (py::ssize_t i = 0; i < trail_input.shape(0); ++i) {
            const int value = trail_input(i);
            if (value < 0 || value >= n_) {
                throw std::out_of_range("trail node outside graph");
            }
            trails_[static_cast<std::size_t>(i)] = value;
        }
        int previous = 0;
        for (py::ssize_t i = 0; i < break_input.shape(0); ++i) {
            const int value = break_input(i);
            if (value <= previous || value > static_cast<int>(trails_.size())) {
                throw std::invalid_argument("invalid breakpoints");
            }
            breakpoints_[static_cast<std::size_t>(i)] = value;
            previous = value;
        }
        ntrails_ = static_cast<int>(breakpoints_.size());
        nz_ = breakpoints_.back();
        if (nz_ != static_cast<int>(trails_.size())) {
            throw std::invalid_argument("last breakpoint must equal trail size");
        }

        nzmap_.assign(static_cast<std::size_t>(n_), 0);
        zmap_.assign(static_cast<std::size_t>(nz_), 0);
        std::vector<int> offsets(static_cast<std::size_t>(n_), 0);
        std::vector<int> prefix(static_cast<std::size_t>(n_), 0);
        for (int node : trails_) ++nzmap_[static_cast<std::size_t>(node)];
        prefix[0] = offsets[0] = nzmap_[0];
        for (int i = 1; i < n_; ++i) {
            prefix[static_cast<std::size_t>(i)] =
                prefix[static_cast<std::size_t>(i - 1)] +
                nzmap_[static_cast<std::size_t>(i)];
            offsets[static_cast<std::size_t>(i)] =
                nzmap_[static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < nz_; ++i) {
            const int node = trails_[static_cast<std::size_t>(i)];
            const int mapped =
                prefix[static_cast<std::size_t>(node)] -
                offsets[static_cast<std::size_t>(node)];
            zmap_[static_cast<std::size_t>(mapped)] = i;
            --offsets[static_cast<std::size_t>(node)];
        }

        std::vector<int> degrees(static_cast<std::size_t>(n_), 0);
        int trail_start = 0;
        for (int trail = 0; trail < ntrails_; ++trail) {
            const int trail_end =
                breakpoints_[static_cast<std::size_t>(trail)];
            for (int i = trail_start; i + 1 < trail_end; ++i) {
                ++degrees[static_cast<std::size_t>(
                    trails_[static_cast<std::size_t>(i)]
                )];
                ++degrees[static_cast<std::size_t>(
                    trails_[static_cast<std::size_t>(i + 1)]
                )];
            }
            trail_start = trail_end;
        }
        adjacency_offsets_.resize(static_cast<std::size_t>(n_ + 1), 0);
        for (int i = 0; i < n_; ++i) {
            adjacency_offsets_[static_cast<std::size_t>(i + 1)] =
                adjacency_offsets_[static_cast<std::size_t>(i)] +
                degrees[static_cast<std::size_t>(i)];
        }
        adjacency_.resize(
            static_cast<std::size_t>(adjacency_offsets_.back())
        );
        offsets = adjacency_offsets_;
        trail_start = 0;
        for (int trail = 0; trail < ntrails_; ++trail) {
            const int trail_end =
                breakpoints_[static_cast<std::size_t>(trail)];
            for (int i = trail_start; i + 1 < trail_end; ++i) {
                const int left = trails_[static_cast<std::size_t>(i)];
                const int right = trails_[static_cast<std::size_t>(i + 1)];
                adjacency_[static_cast<std::size_t>(
                    offsets[static_cast<std::size_t>(left)]++
                )] = right;
                adjacency_[static_cast<std::size_t>(
                    offsets[static_cast<std::size_t>(right)]++
                )] = left;
            }
            trail_start = trail_end;
        }

        maximum_trail_ = breakpoints_[0];
        for (int i = 1; i < ntrails_; ++i) {
            maximum_trail_ = std::max(
                maximum_trail_,
                breakpoints_[static_cast<std::size_t>(i)] -
                    breakpoints_[static_cast<std::size_t>(i - 1)]
            );
        }
        num_threads_ = std::min(num_threads_, ntrails_);
        zold_.resize(static_cast<std::size_t>(nz_));
        tf_buffer_.resize(
            static_cast<std::size_t>(nz_ * 8 - 2 * ntrails_)
        );
        y_buffer_.resize(
            static_cast<std::size_t>(maximum_trail_ * num_threads_)
        );
        w_buffer_.resize(
            static_cast<std::size_t>(maximum_trail_ * num_threads_)
        );
    }

    int solve(
        const py::array_t<double, py::array::c_style>& y,
        const py::array_t<double, py::array::c_style>& weights,
        py::array_t<double, py::array::c_style>& beta,
        py::array_t<double, py::array::c_style>& z_state,
        py::array_t<double, py::array::c_style>& u,
        double lambda,
        double alpha = 2.0,
        double inflate = 2.0,
        int maxsteps = 100000,
        double converge = 1e-6
    ) {
        if (y.size() != n_ || weights.size() != n_ || beta.size() != n_ ||
                z_state.size() != nz_ || u.size() != nz_) {
            throw std::invalid_argument("state array has an invalid length");
        }
        if (!(lambda >= 0.0) || !(alpha > 0.0) || !(inflate > 0.0) ||
                maxsteps <= 0 || !(converge > 0.0)) {
            throw std::invalid_argument("invalid solver parameter");
        }
        auto y_view = y.unchecked<1>();
        auto weight_view = weights.unchecked<1>();
        auto beta_view = beta.mutable_unchecked<1>();
        auto z_view = z_state.mutable_unchecked<1>();
        auto u_view = u.mutable_unchecked<1>();
        double* y_ptr = const_cast<double*>(&y_view(0));
        double* weight_ptr = const_cast<double*>(&weight_view(0));
        double* beta_ptr = &beta_view(0);
        double* z_input = &z_view(0);
        double* u_ptr = &u_view(0);

        std::memcpy(
            zold_.data(), z_input,
            static_cast<std::size_t>(nz_) * sizeof(double)
        );
        std::fill(w_buffer_.begin(), w_buffer_.end(), alpha / 2.0);
        double* z_current = z_input;
        double* z_previous = zold_.data();
        const double* original_z = z_input;
        int step = 0;
        double current_convergence = converge + 1.0;
        {
            py::gil_scoped_release release;
            while (step < maxsteps && current_convergence > converge) {
                update_beta_weight(
                    n_, y_ptr, weight_ptr, z_current, u_ptr,
                    nzmap_.data(), zmap_.data(), alpha, beta_ptr
                );
                std::swap(z_current, z_previous);
                update_z_workspace(beta_ptr, u_ptr, lambda, z_current);
                update_u(
                    n_, beta_ptr, z_current, zmap_.data(),
                    nzmap_.data(), u_ptr
                );
                const double primal = primal_resnorm(
                    n_, beta_ptr, z_current, nzmap_.data(), zmap_.data()
                );
                const double dual = dual_resnorm(
                    nz_, z_current, z_previous, alpha
                );
                current_convergence = std::max(primal, dual);
                if (step % VARYING_PENALTY_DELAY == 0 &&
                        primal > 10.0 * dual) {
                    alpha *= inflate;
                    for (int i = 0; i < nz_; ++i) u_ptr[i] /= inflate;
                    std::fill(
                        w_buffer_.begin(), w_buffer_.end(), alpha / 2.0
                    );
                } else if (step % VARYING_PENALTY_DELAY == 0 &&
                           dual > 10.0 * primal) {
                    alpha /= inflate;
                    for (int i = 0; i < nz_; ++i) u_ptr[i] *= inflate;
                    std::fill(
                        w_buffer_.begin(), w_buffer_.end(), alpha / 2.0
                    );
                }
                ++step;
            }
            // The released C routine only returns z_state when the final
            // alternating buffer is the caller-owned pointer. Preserve that
            // behavior exactly for warm-start parity.
            if (z_current == original_z) {
                // No copy is needed: caller-owned z_state already is final.
            }
        }
        return step;
    }

    int nnodes() const { return n_; }
    int nz() const { return nz_; }
    int ntrails() const { return ntrails_; }
    int num_threads() const { return num_threads_; }

    int plateau_count(
        const py::array_t<double, py::array::c_style>& beta,
        double tolerance = 1e-4
    ) const {
        if (beta.size() != n_ || !(tolerance >= 0.0)) {
            throw std::invalid_argument("invalid beta or tolerance");
        }
        const auto values = beta.unchecked<1>();
        std::vector<unsigned char> checked(
            static_cast<std::size_t>(n_), 0
        );
        std::vector<int> queue(static_cast<std::size_t>(n_));
        for (int i = 0; i < n_; ++i) {
            if (std::isnan(values(i))) {
                checked[static_cast<std::size_t>(i)] = 1;
            }
        }
        int count = 0;
        for (int start = 0; start < n_; ++start) {
            if (checked[static_cast<std::size_t>(start)]) continue;
            ++count;
            const double lower = values(start) - tolerance;
            const double upper = values(start) + tolerance;
            int head = 0;
            int tail = 1;
            queue[0] = start;
            while (head < tail) {
                const int node = queue[static_cast<std::size_t>(head++)];
                const int begin =
                    adjacency_offsets_[static_cast<std::size_t>(node)];
                const int end =
                    adjacency_offsets_[static_cast<std::size_t>(node + 1)];
                for (int offset = begin; offset < end; ++offset) {
                    const int neighbor =
                        adjacency_[static_cast<std::size_t>(offset)];
                    if (!checked[static_cast<std::size_t>(neighbor)] &&
                            values(neighbor) >= lower &&
                            values(neighbor) <= upper) {
                        checked[static_cast<std::size_t>(neighbor)] = 1;
                        queue[static_cast<std::size_t>(tail++)] = neighbor;
                    }
                }
            }
        }
        return count;
    }

private:
    void update_z_workspace(
        const double* beta,
        const double* u,
        double lambda,
        double* z
    ) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads_)
#endif
        for (int trail = 0; trail < ntrails_; ++trail) {
            const int trail_start =
                trail == 0
                    ? 0
                    : breakpoints_[static_cast<std::size_t>(trail - 1)];
            const int trail_end =
                breakpoints_[static_cast<std::size_t>(trail)];
            const int trail_size = trail_end - trail_start;
#ifdef _OPENMP
            const int thread = omp_get_thread_num();
#else
            const int thread = 0;
#endif
            double* y_buffer =
                y_buffer_.data() +
                static_cast<std::size_t>(thread * maximum_trail_);
            double* w_buffer =
                w_buffer_.data() +
                static_cast<std::size_t>(thread * maximum_trail_);
            for (int offset = 0; offset < trail_size; ++offset) {
                const int position = trail_start + offset;
                y_buffer[offset] =
                    beta[trails_[static_cast<std::size_t>(position)]] +
                    u[position];
            }
            double* x =
                tf_buffer_.data() +
                static_cast<std::size_t>(8 * trail_start - 2 * trail);
            double* a = x + 2 * trail_size;
            double* b = x + 4 * trail_size;
            double* tm = x + 6 * trail_size;
            double* tp = x + 7 * trail_size - 1;
            tf_dp_weight(
                trail_size, y_buffer, w_buffer, lambda, z + trail_start,
                x, a, b, tm, tp
            );
        }
    }

    int n_;
    int ntrails_;
    int nz_;
    int num_threads_;
    int maximum_trail_;
    std::vector<int> trails_;
    std::vector<int> breakpoints_;
    std::vector<int> nzmap_;
    std::vector<int> zmap_;
    std::vector<int> adjacency_offsets_;
    std::vector<int> adjacency_;
    std::vector<double> zold_;
    std::vector<double> tf_buffer_;
    std::vector<double> y_buffer_;
    std::vector<double> w_buffer_;
};

class GridTVWorkspace {
public:
    GridTVWorkspace(
        const py::array_t<
            int, py::array::c_style | py::array::forcecast>& dims,
        int num_threads = 1
    ) : num_threads_(num_threads) {
        if (dims.size() != 3 || num_threads_ <= 0) {
            throw std::invalid_argument(
                "dims must have length three and num_threads must be positive"
            );
        }
        const auto shape = dims.unchecked<1>();
        nx_ = shape(0);
        ny_ = shape(1);
        nz_ = shape(2);
        if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) {
            throw std::invalid_argument("grid dimensions must be positive");
        }
        n_ = nx_ * ny_ * nz_;
        beta_bar_.resize(static_cast<std::size_t>(n_));
        beta_old_.resize(static_cast<std::size_t>(n_));
    }

    int solve(
        const py::array_t<double, py::array::c_style>& y,
        const py::array_t<double, py::array::c_style>& weights,
        py::array_t<double, py::array::c_style>& beta,
        py::array_t<double, py::array::c_style>& dual,
        double lambda,
        int maxsteps = 400,
        double converge = 1e-5,
        double tau = 0.25,
        double sigma = 0.25
    ) {
        if (y.size() != n_ || weights.size() != n_ || beta.size() != n_ ||
                dual.size() != 3 * n_) {
            throw std::invalid_argument("grid-TV state array has bad length");
        }
        if (!(lambda >= 0.0) || maxsteps <= 0 || !(converge > 0.0) ||
                !(tau > 0.0) || !(sigma > 0.0) ||
                tau * sigma * 12.0 >= 1.0) {
            throw std::invalid_argument("invalid grid-TV solver parameter");
        }
        const auto y_view = y.unchecked<1>();
        const auto weight_view = weights.unchecked<1>();
        auto beta_view = beta.mutable_unchecked<1>();
        auto dual_view = dual.mutable_unchecked<1>();
        const double* y_ptr = &y_view(0);
        const double* weight_ptr = &weight_view(0);
        double* beta_ptr = &beta_view(0);
        double* px = &dual_view(0);
        double* py_axis = px + n_;
        double* pz = py_axis + n_;
        std::copy(beta_ptr, beta_ptr + n_, beta_bar_.begin());

        int completed = 0;
        {
            py::gil_scoped_release release;
            for (int iteration = 1; iteration <= maxsteps; ++iteration) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads_)
#endif
                for (int index = 0; index < n_; ++index) {
                    const int z = index % nz_;
                    const int yz = index / nz_;
                    const int y_index = yz % ny_;
                    const int x = yz / ny_;
                    px[index] = x + 1 < nx_
                        ? std::clamp(
                            px[index] + sigma * (
                                beta_bar_[static_cast<std::size_t>(
                                    index + ny_ * nz_
                                )] -
                                beta_bar_[static_cast<std::size_t>(index)]
                            ),
                            -lambda, lambda
                        )
                        : 0.0;
                    py_axis[index] = y_index + 1 < ny_
                        ? std::clamp(
                            py_axis[index] + sigma * (
                                beta_bar_[static_cast<std::size_t>(
                                    index + nz_
                                )] -
                                beta_bar_[static_cast<std::size_t>(index)]
                            ),
                            -lambda, lambda
                        )
                        : 0.0;
                    pz[index] = z + 1 < nz_
                        ? std::clamp(
                            pz[index] + sigma * (
                                beta_bar_[static_cast<std::size_t>(index + 1)] -
                                beta_bar_[static_cast<std::size_t>(index)]
                            ),
                            -lambda, lambda
                        )
                        : 0.0;
                }

                std::copy(beta_ptr, beta_ptr + n_, beta_old_.begin());
                double maximum_change = 0.0;
                double maximum_beta = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads_) \
    reduction(max:maximum_change,maximum_beta)
#endif
                for (int index = 0; index < n_; ++index) {
                    const int z = index % nz_;
                    const int yz = index / nz_;
                    const int y_index = yz % ny_;
                    const int x = yz / ny_;
                    double adjoint = -px[index] - py_axis[index] - pz[index];
                    if (x > 0) adjoint += px[index - ny_ * nz_];
                    if (y_index > 0) adjoint += py_axis[index - nz_];
                    if (z > 0) adjoint += pz[index - 1];
                    const double old_value =
                        beta_old_[static_cast<std::size_t>(index)];
                    const double new_value = (
                        old_value - tau * adjoint +
                        tau * weight_ptr[index] * y_ptr[index]
                    ) / (1.0 + tau * weight_ptr[index]);
                    beta_ptr[index] = new_value;
                    beta_bar_[static_cast<std::size_t>(index)] =
                        2.0 * new_value - old_value;
                    maximum_change = std::max(
                        maximum_change, std::abs(new_value - old_value)
                    );
                    maximum_beta =
                        std::max(maximum_beta, std::abs(new_value));
                }
                completed = iteration;
                if (iteration % 10 == 0 &&
                        maximum_change /
                            std::max(1.0, maximum_beta) < converge) {
                    break;
                }
            }
        }
        return completed;
    }

    int plateau_count(
        const py::array_t<double, py::array::c_style>& beta,
        double tolerance = 1e-4
    ) const {
        if (beta.size() != n_ || !(tolerance >= 0.0)) {
            throw std::invalid_argument("invalid beta or tolerance");
        }
        const auto values = beta.unchecked<1>();
        std::vector<unsigned char> checked(
            static_cast<std::size_t>(n_), 0
        );
        std::vector<int> queue(static_cast<std::size_t>(n_));
        int count = 0;
        for (int start = 0; start < n_; ++start) {
            if (checked[static_cast<std::size_t>(start)]) continue;
            ++count;
            const double lower = values(start) - tolerance;
            const double upper = values(start) + tolerance;
            int head = 0;
            int tail = 1;
            checked[static_cast<std::size_t>(start)] = 1;
            queue[0] = start;
            while (head < tail) {
                const int node = queue[static_cast<std::size_t>(head++)];
                const int z = node % nz_;
                const int yz = node / nz_;
                const int y_index = yz % ny_;
                const int x = yz / ny_;
                const int neighbors[6] = {
                    x > 0 ? node - ny_ * nz_ : -1,
                    x + 1 < nx_ ? node + ny_ * nz_ : -1,
                    y_index > 0 ? node - nz_ : -1,
                    y_index + 1 < ny_ ? node + nz_ : -1,
                    z > 0 ? node - 1 : -1,
                    z + 1 < nz_ ? node + 1 : -1
                };
                for (const int neighbor : neighbors) {
                    if (neighbor >= 0 &&
                            !checked[static_cast<std::size_t>(neighbor)] &&
                            values(neighbor) >= lower &&
                            values(neighbor) <= upper) {
                        checked[static_cast<std::size_t>(neighbor)] = 1;
                        queue[static_cast<std::size_t>(tail++)] = neighbor;
                    }
                }
            }
        }
        return count;
    }

    int nnodes() const { return n_; }
    int num_threads() const { return num_threads_; }

private:
    int nx_;
    int ny_;
    int nz_;
    int n_;
    int num_threads_;
    std::vector<double> beta_bar_;
    std::vector<double> beta_old_;
};

PYBIND11_MODULE(persistent_gfl_backend, module) {
    module.doc() = "Persistent-workspace wrapper around released libgraphfl";
    py::class_<PersistentGFLWorkspace>(module, "PersistentGFLWorkspace")
        .def(py::init<
             int,
             const py::array_t<int, py::array::c_style | py::array::forcecast>&,
             const py::array_t<int, py::array::c_style | py::array::forcecast>&,
             int>(),
             py::arg("nnodes"), py::arg("trails"),
             py::arg("breakpoints"), py::arg("num_threads") = 1)
        .def(
            "solve", &PersistentGFLWorkspace::solve,
            py::arg("y"), py::arg("weights"), py::arg("beta"),
            py::arg("z"), py::arg("u"), py::arg("lambda"),
            py::arg("alpha") = 2.0, py::arg("inflate") = 2.0,
            py::arg("maxsteps") = 100000, py::arg("converge") = 1e-6
        )
        .def_property_readonly("nnodes", &PersistentGFLWorkspace::nnodes)
        .def_property_readonly("nz", &PersistentGFLWorkspace::nz)
        .def_property_readonly("ntrails", &PersistentGFLWorkspace::ntrails)
        .def_property_readonly(
            "num_threads", &PersistentGFLWorkspace::num_threads
        )
        .def(
            "plateau_count", &PersistentGFLWorkspace::plateau_count,
            py::arg("beta"), py::arg("tolerance") = 1e-4
        );
    py::class_<GridTVWorkspace>(module, "GridTVWorkspace")
        .def(
            py::init<
                const py::array_t<
                    int,
                    py::array::c_style | py::array::forcecast>&,
                int>(),
            py::arg("dims"), py::arg("num_threads") = 1
        )
        .def(
            "solve", &GridTVWorkspace::solve,
            py::arg("y"), py::arg("weights"), py::arg("beta"),
            py::arg("dual"), py::arg("lambda"),
            py::arg("maxsteps") = 400, py::arg("converge") = 1e-5,
            py::arg("tau") = 0.25, py::arg("sigma") = 0.25
        )
        .def_property_readonly("nnodes", &GridTVWorkspace::nnodes)
        .def_property_readonly(
            "num_threads", &GridTVWorkspace::num_threads
        )
        .def(
            "plateau_count", &GridTVWorkspace::plateau_count,
            py::arg("beta"), py::arg("tolerance") = 1e-4
        );
}


