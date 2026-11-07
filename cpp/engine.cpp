// C++ backtest engine for the Calendar rebalancing strategy.
//
// Mirrors the Python reference implementation in
// src/rebalancing_frontrun/signals/calendar.py and backtest/strategy.py:
//   - 60/40 ES/ZN drift signal, reset after each month-end close
//   - positions set at the previous close (no look-ahead), sign of the signal only
//   - each leg sized to a target annualized vol from a trailing window
//   - per-side costs on the daily change in each leg's weight
//
// It also runs the placebo test: the same trade on a randomly placed block of
// `window` trading days inside each month, repeated n_sims times. Speed matters
// there (thousands of full backtests), which is why this engine exists.
//
// Conventions shared with the Python code:
//   - NaN returns (market closures) count as 0 for that leg
//   - days_to_month_end is 0 on the last trading day of a month, NaN in an
//     incomplete final month
//   - standard deviations use ddof = 1, annualized with sqrt(252)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Vec = std::vector<double>;
using Mask = std::vector<char>;
using InArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kTradingDays = 252.0;
constexpr int kVolMinPeriods = 20;

// ---------- conversions ----------

Vec to_vec(const InArray& a) {
    if (a.ndim() != 1) throw std::invalid_argument("expected a 1-D array");
    auto r = a.unchecked<1>();
    Vec v(static_cast<size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) v[i] = r(i);
    return v;
}

py::array_t<double> to_array(const Vec& v) {
    return py::array_t<double>(static_cast<py::ssize_t>(v.size()), v.data());
}

py::array_t<bool> to_bool_array(const Mask& m) {
    py::array_t<bool> out(static_cast<py::ssize_t>(m.size()));
    auto w = out.mutable_unchecked<1>();
    for (size_t i = 0; i < m.size(); ++i) w(i) = m[i] != 0;
    return out;
}

// ---------- building blocks ----------

Vec fill_zero(const Vec& r) {
    Vec out(r);
    for (double& x : out)
        if (std::isnan(x)) x = 0.0;
    return out;
}

double sign(double x) { return (x > 0.0) - (x < 0.0); }

// Equity weight minus target at each close, before that day's month-end reset.
Vec calendar_signal(const Vec& es, const Vec& zn, const Vec& dtme, double target) {
    Vec out(es.size());
    double w = target;
    for (size_t i = 0; i < es.size(); ++i) {
        const double eq = w * (1.0 + es[i]);
        const double bd = (1.0 - w) * (1.0 + zn[i]);
        w = eq / (eq + bd);
        out[i] = w - target;
        if (dtme[i] == 0.0) w = target;  // rebalance after the month-end close
    }
    return out;
}

// Annualized trailing vol known at the close of day i-1, used on day i.
// Matches pandas: r.rolling(lookback, min_periods=20).std().shift(1) * sqrt(252).
Vec trailing_vol(const Vec& r, int lookback) {
    const size_t n = r.size();
    Vec out(n, kNaN);
    for (size_t i = 1; i < n; ++i) {
        const size_t end = i;  // window covers [start, end)
        const size_t start = end >= static_cast<size_t>(lookback) ? end - lookback : 0;
        const size_t count = end - start;
        if (count < static_cast<size_t>(kVolMinPeriods)) continue;
        double mean = 0.0;
        for (size_t j = start; j < end; ++j) mean += r[j];
        mean /= static_cast<double>(count);
        double ss = 0.0;
        for (size_t j = start; j < end; ++j) ss += (r[j] - mean) * (r[j] - mean);
        out[i] = std::sqrt(ss / static_cast<double>(count - 1)) * std::sqrt(kTradingDays);
    }
    return out;
}

// Days the strategy may hold a position, and the direction multiplier for each.
// timing "holding": the last `window` trading days of each month.
// timing "paper":   the day after each of the last `window` days; front-run on the
//                   last window-1 days, opposite side on the first day of next month.
std::pair<Mask, Vec> window_mask(const Vec& dtme, int window, const std::string& timing) {
    const size_t n = dtme.size();
    Mask in_window(n, 0);
    Vec flip(n, 1.0);
    if (timing == "holding") {
        for (size_t i = 0; i < n; ++i) in_window[i] = dtme[i] < window;  // NaN -> false
    } else if (timing == "paper") {
        for (size_t i = 1; i < n; ++i) {
            const double prev = dtme[i - 1];
            const bool frontrun = prev >= 1.0 && prev < window;
            const bool reversal = prev == 0.0;
            in_window[i] = frontrun || reversal;
            if (reversal) flip[i] = -1.0;
        }
    } else {
        throw std::invalid_argument("timing must be 'holding' or 'paper'");
    }
    return {in_window, flip};
}

struct Params {
    double leg_vol;
    double es_cost;  // fraction of notional per side
    double zn_cost;
};

struct Daily {
    Vec signal_prev, direction, w_es, w_zn, gross, cost, net;
    Mask active;
};

// One full backtest given precomputed signal and vols.
Daily simulate(const Vec& es0, const Vec& zn0, const Vec& signal, const Vec& vol_es,
               const Vec& vol_zn, const Mask& in_window, const Vec& flip, const Params& p) {
    const size_t n = es0.size();
    Daily d;
    for (Vec* v : {&d.signal_prev, &d.direction, &d.w_es, &d.w_zn, &d.gross, &d.cost, &d.net})
        v->assign(n, 0.0);
    d.active.assign(n, 0);

    double prev_es = 0.0, prev_zn = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double sp = i > 0 ? signal[i - 1] : kNaN;
        d.signal_prev[i] = sp;
        const bool active = in_window[i] && !std::isnan(sp) && !std::isnan(vol_es[i]) &&
                            !std::isnan(vol_zn[i]);
        d.active[i] = active;
        if (active) {
            const double dir = -sign(sp) * flip[i];
            d.direction[i] = dir;
            d.w_es[i] = dir * p.leg_vol / vol_es[i];
            d.w_zn[i] = -dir * p.leg_vol / vol_zn[i];
        }
        d.gross[i] = d.w_es[i] * es0[i] + d.w_zn[i] * zn0[i];
        d.cost[i] = std::abs(d.w_es[i] - prev_es) * p.es_cost +
                    std::abs(d.w_zn[i] - prev_zn) * p.zn_cost;
        d.net[i] = d.gross[i] - d.cost[i];
        prev_es = d.w_es[i];
        prev_zn = d.w_zn[i];
    }
    return d;
}

// Lean version for the placebo loop: net P&L only, no per-day bookkeeping.
void simulate_net(const Vec& es0, const Vec& zn0, const Vec& signal, const Vec& vol_es,
                  const Vec& vol_zn, const Mask& in_window, const Params& p, Vec& net) {
    double prev_es = 0.0, prev_zn = 0.0;
    for (size_t i = 0; i < es0.size(); ++i) {
        double w_es = 0.0, w_zn = 0.0;
        if (i > 0 && in_window[i] && !std::isnan(vol_es[i]) && !std::isnan(vol_zn[i])) {
            const double dir = -sign(signal[i - 1]);
            w_es = dir * p.leg_vol / vol_es[i];
            w_zn = -dir * p.leg_vol / vol_zn[i];
        }
        const double cost =
            std::abs(w_es - prev_es) * p.es_cost + std::abs(w_zn - prev_zn) * p.zn_cost;
        net[i] = w_es * es0[i] + w_zn * zn0[i] - cost;
        prev_es = w_es;
        prev_zn = w_zn;
    }
}

// Annualized Sharpe of daily P&L over [begin, end), ddof = 1. NaN if undefined.
double sharpe(const Vec& pnl, size_t begin, size_t end) {
    const size_t count = end - begin;
    if (count < 2) return kNaN;
    double mean = 0.0;
    for (size_t i = begin; i < end; ++i) mean += pnl[i];
    mean /= static_cast<double>(count);
    double ss = 0.0;
    for (size_t i = begin; i < end; ++i) ss += (pnl[i] - mean) * (pnl[i] - mean);
    const double sd = std::sqrt(ss / static_cast<double>(count - 1));
    return sd > 0.0 ? mean / sd * std::sqrt(kTradingDays) : kNaN;
}

// Complete months as [first, last] index pairs. A month is complete when it
// ends on a day with days_to_month_end == 0; the incomplete final month is skipped.
std::vector<std::pair<size_t, size_t>> month_blocks(const Vec& dtme) {
    std::vector<std::pair<size_t, size_t>> blocks;
    size_t start = 0;
    for (size_t i = 0; i < dtme.size(); ++i) {
        if (dtme[i] == 0.0) {
            blocks.emplace_back(start, i);
            start = i + 1;
        }
    }
    return blocks;
}

void check_lengths(const Vec& a, const Vec& b, const Vec& c) {
    if (a.size() != b.size() || a.size() != c.size())
        throw std::invalid_argument("es_ret, zn_ret and days_to_month_end must have equal length");
}

// ---------- Python-facing functions ----------

py::dict py_backtest(const InArray& es_ret, const InArray& zn_ret, const InArray& dtme_in,
                     int window, int vol_lookback, double leg_vol, double es_cost_bps,
                     double zn_cost_bps, const std::string& timing) {
    const Vec es0 = fill_zero(to_vec(es_ret));
    const Vec zn0 = fill_zero(to_vec(zn_ret));
    const Vec dtme = to_vec(dtme_in);
    check_lengths(es0, zn0, dtme);

    const Vec signal = calendar_signal(es0, zn0, dtme, 0.60);
    const Vec vol_es = trailing_vol(es0, vol_lookback);
    const Vec vol_zn = trailing_vol(zn0, vol_lookback);
    auto [in_window, flip] = window_mask(dtme, window, timing);
    const Params p{leg_vol, es_cost_bps / 1e4, zn_cost_bps / 1e4};

    const Daily d = simulate(es0, zn0, signal, vol_es, vol_zn, in_window, flip, p);
    py::dict out;
    out["signal_prev"] = to_array(d.signal_prev);
    out["active"] = to_bool_array(d.active);
    out["direction"] = to_array(d.direction);
    out["w_es"] = to_array(d.w_es);
    out["w_zn"] = to_array(d.w_zn);
    out["gross"] = to_array(d.gross);
    out["cost"] = to_array(d.cost);
    out["net"] = to_array(d.net);
    return out;
}

py::dict py_placebo(const InArray& es_ret, const InArray& zn_ret, const InArray& dtme_in,
                    int n_sims, std::uint64_t seed, size_t split, int window, int vol_lookback,
                    double leg_vol, double es_cost_bps, double zn_cost_bps) {
    const Vec es0 = fill_zero(to_vec(es_ret));
    const Vec zn0 = fill_zero(to_vec(zn_ret));
    const Vec dtme = to_vec(dtme_in);
    check_lengths(es0, zn0, dtme);
    const size_t n = es0.size();
    if (split > n) throw std::invalid_argument("split must be <= number of days");

    const Vec signal = calendar_signal(es0, zn0, dtme, 0.60);
    const Vec vol_es = trailing_vol(es0, vol_lookback);
    const Vec vol_zn = trailing_vol(zn0, vol_lookback);
    const Params p{leg_vol, es_cost_bps / 1e4, zn_cost_bps / 1e4};
    const auto blocks = month_blocks(dtme);

    std::mt19937_64 rng(seed);
    Vec full(n_sims), in_sample(n_sims), out_sample(n_sims);
    Mask in_window(n);
    Vec net(n);

    {
        py::gil_scoped_release release;  // pure C++ from here on
        for (int s = 0; s < n_sims; ++s) {
            std::fill(in_window.begin(), in_window.end(), 0);
            for (const auto& [first, last] : blocks) {
                const size_t len = last - first + 1;
                if (len < static_cast<size_t>(window)) continue;
                std::uniform_int_distribution<size_t> pick(first, last + 1 - window);
                const size_t start = pick(rng);
                for (size_t i = start; i < start + window; ++i) in_window[i] = 1;
            }
            simulate_net(es0, zn0, signal, vol_es, vol_zn, in_window, p, net);
            full[s] = sharpe(net, 0, n);
            in_sample[s] = sharpe(net, 0, split);
            out_sample[s] = sharpe(net, split, n);
        }
    }

    py::dict out;
    out["sharpe_full"] = to_array(full);
    out["sharpe_in_sample"] = to_array(in_sample);
    out["sharpe_out_of_sample"] = to_array(out_sample);
    return out;
}

}  // namespace

PYBIND11_MODULE(_engine, m) {
    m.doc() = "C++ backtest engine for the Calendar rebalancing strategy";

    m.def("version", []() { return "0.2.0"; }, "Engine version, used to check the build.");

    m.def("backtest", &py_backtest,
          "Run one backtest. Returns daily arrays: signal_prev, active, direction, "
          "w_es, w_zn, gross, cost, net.",
          py::arg("es_ret"), py::arg("zn_ret"), py::arg("days_to_month_end"),
          py::arg("window") = 5, py::arg("vol_lookback") = 60, py::arg("leg_vol") = 0.10,
          py::arg("es_cost_bps") = 0.5, py::arg("zn_cost_bps") = 1.5,
          py::arg("timing") = "holding");

    m.def("placebo", &py_placebo,
          "Run n_sims backtests, each holding a randomly placed block of `window` trading "
          "days in every complete month. Returns net Sharpe ratios for the full sample, "
          "days [0, split) and days [split, end).",
          py::arg("es_ret"), py::arg("zn_ret"), py::arg("days_to_month_end"),
          py::arg("n_sims") = 10000, py::arg("seed") = 0, py::arg("split") = 0,
          py::arg("window") = 5, py::arg("vol_lookback") = 60, py::arg("leg_vol") = 0.10,
          py::arg("es_cost_bps") = 0.5, py::arg("zn_cost_bps") = 1.5);
}
