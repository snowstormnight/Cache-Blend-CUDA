// loading_controller.cpp
//
// Overview:
//   Implements all logic declared in loading_controller.h.
//   Provides delay estimators, binary search to find crossover recompute ratio, and quality prediction.
//
// Parts:
//   Part 1: Includes and internal helpers.
//   Part 2: Delay estimation implementations.
//   Part 3: Crossover search and output assembly.

#include "loading_controller.h"
#include <algorithm>
#include <iostream>

//////////////////////
// Part 1: Helpers   //
//////////////////////

// (No additional free helpers outside class needed for this version)

///////////////////////////////
// Part 2: Delay Estimators   //
///////////////////////////////

double LoadingController::EstimateLoadingDelayMs(const ModelConfig& model, int context_length, const StorageDevice& storage) const {
    // Compute KV cache byte size: context_length * num_layers * 2 (K + V) * hidden_dim * element_size
    double element_size = ElementSizeBytes(model.quant_level);
    double total_bytes = double(context_length) * model.num_layers * 2.0 * model.hidden_dim * element_size;
    double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
    double transfer_time_s = total_gb / storage.bandwidth_gbps;
    double transfer_time_ms = transfer_time_s * 1000.0;
    double coordination_overhead = model.num_layers * storage.per_layer_overhead_ms;
    double total_delay = transfer_time_ms + storage.latency_ms + coordination_overhead;
    return total_delay;
}

double LoadingController::EstimateRecomputeDelayMs(double full_prefill_time_ms, double ratio) const {
    return ratio * full_prefill_time_ms * (1.0 + overhead_factor_);
}

double LoadingController::PredictQualityScore(double ratio) const {
    // Placeholder quality model: linear interpolation from floor to 1.0
    double floor = 0.8;
    if (ratio <= min_quality_ratio_) return floor;
    double span = 1.0 - floor;
    double normalized = (ratio - min_quality_ratio_) / (1.0 - min_quality_ratio_);
    return floor + normalized * span;
}

/////////////////////////////////////
// Part 3: Crossover & Main API     //
/////////////////////////////////////

double LoadingController::FindCrossoverRatio(double full_prefill_time_ms,
                                             const ModelConfig& model,
                                             int context_length,
                                             const StorageDevice& storage) const {
    double low = min_quality_ratio_;
    double high = 1.0;
    double best = low;
    // Binary search until difference under 0.1% (0.001 absolute)
    while (high - low > 0.001) {
        double mid = (low + high) / 2.0;
        double trecompute = EstimateRecomputeDelayMs(full_prefill_time_ms, mid);
        double tload = EstimateLoadingDelayMs(model, context_length, storage);
        if (trecompute <= tload) {
            best = mid;
            low = mid;
        } else {
            high = mid;
        }
    }
    return std::max(best, min_quality_ratio_);
}

LoadingControllerOutput LoadingController::ComputeOptimalRatio(
    const ModelConfig& model,
    int context_length,
    const StorageDevice& storage,
    const PrefillProfile& profile
) {
    LoadingControllerOutput out{};
    auto maybe_full = profile.get_prefill_time(model.name, context_length);
    if (!maybe_full.has_value()) {
        std::cerr << "[LoadingController] Missing prefill profile for model/context. Using minimum ratio fallback.\n";
        out.optimal_recompute_ratio = min_quality_ratio_;
        out.quality_score = PredictQualityScore(out.optimal_recompute_ratio);
        double recompute_delay = EstimateRecomputeDelayMs(1.0 /* fallback */, out.optimal_recompute_ratio);
        double load_delay = EstimateLoadingDelayMs(model, context_length, storage);
        out.achieves_hiding = recompute_delay <= load_delay;
        out.expected_total_delay_ms = std::max(load_delay, recompute_delay);
        return out;
    }

    double full_prefill_time_ms = maybe_full.value();
    double crossover = FindCrossoverRatio(full_prefill_time_ms, model, context_length, storage);
    double recompute_delay = EstimateRecomputeDelayMs(full_prefill_time_ms, crossover);
    double load_delay = EstimateLoadingDelayMs(model, context_length, storage);
    out.optimal_recompute_ratio = crossover;
    out.achieves_hiding = recompute_delay <= load_delay;
    out.expected_total_delay_ms = std::max(recompute_delay, load_delay);
    out.quality_score = PredictQualityScore(out.optimal_recompute_ratio);
    return out;
}
