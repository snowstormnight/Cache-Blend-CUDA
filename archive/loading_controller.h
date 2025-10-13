// loading_controller.h
//
// Overview:
//   Implements Part 1 of the controller: recompute ratio optimization given a fixed storage device.
//   Given model/config/context and a specific storage device, it finds the maximum recompute ratio
//   such that selective recomputation can be hidden behind KV cache loading (i.e., no extra delay),
//   subject to a minimum quality floor. Outputs include ratio, hiding flag, expected delay, and quality score.
//
// Parts:
//   Part 1: Types and Data Structures (ModelConfig, StorageDevice, PrefillProfile, Output)
//   Part 2: LoadingController class interface with exposed helpers and primary API.

#pragma once
#include <string>
#include <optional>
#include <map>

///////////////////////
// Part 1: Types / Data //
///////////////////////

enum class QuantizationLevel {
    FP16,
    INT8,
    INT4
};

struct ModelConfig {
    std::string name;
    int num_layers;
    int hidden_dim;      // e.g., 4096
    int num_heads;
    QuantizationLevel quant_level;
};

struct StorageDevice {
    std::string name;
    // Bandwidth in GB/s
    double bandwidth_gbps;
    // Latency in milliseconds
    double latency_ms;
    // Coordination overhead per layer in milliseconds
    double per_layer_overhead_ms;
};

struct PrefillProfile {
    // key is (model name, context length)
    std::map<std::pair<std::string, int>, double> full_prefill_time_ms;

    // Lookup full prefill time (ms). Returns std::nullopt if missing.
    std::optional<double> get_prefill_time(const std::string& model_name, int context_length) const {
        auto key = std::make_pair(model_name, context_length);
        auto it = full_prefill_time_ms.find(key);
        if (it == full_prefill_time_ms.end()) return std::nullopt;
        return it->second;
    }
};

struct LoadingControllerOutput {
    double optimal_recompute_ratio; // [0,1]
    bool achieves_hiding;
    double expected_total_delay_ms;
    double quality_score; // predicted quality metric (normalized)
};

/////////////////////////////
// Part 2: Controller Class //
/////////////////////////////

class LoadingController {
public:
    // Constructor parameters:
    //   min_quality_ratio: empirical floor (e.g., 0.15) below which quality degrades.
    //   overhead_factor: extra cost factor for selective recompute mechanism (e.g., 5% -> 0.05).
    LoadingController(double min_quality_ratio = 0.15,
                      double overhead_factor = 0.05)
        : min_quality_ratio_(min_quality_ratio),
          overhead_factor_(overhead_factor) {}

    // Compute optimal recompute ratio for a given model/context and fixed storage device.
    LoadingControllerOutput ComputeOptimalRatio(
        const ModelConfig& model,
        int context_length,
        const StorageDevice& storage,
        const PrefillProfile& profile
    );

    // Exposed helpers for reuse in other modules (e.g., StorageSelector)
    double EstimateLoadingDelayMs(const ModelConfig& model, int context_length, const StorageDevice& storage) const;
    double EstimateRecomputeDelayMs(double full_prefill_time_ms, double ratio) const;
    double PredictQualityScore(double ratio) const;

private:
    // Binary search to find crossover ratio where recompute delay ≈ load delay
    double FindCrossoverRatio(double full_prefill_time_ms,
                              const ModelConfig& model,
                              int context_length,
                              const StorageDevice& storage) const;

    double min_quality_ratio_;
    double overhead_factor_;

    // Helper to convert quantization level to element size in bytes
    static double ElementSizeBytes(QuantizationLevel q) {
        switch (q) {
            case QuantizationLevel::FP16: return 2.0;
            case QuantizationLevel::INT8: return 1.0;
            case QuantizationLevel::INT4: return 0.5; // approximate
        }
        return 2.0;
    }
};
