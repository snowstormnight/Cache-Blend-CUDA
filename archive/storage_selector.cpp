// storage_selector.cpp
//
// Overview:
//   Implements logic for storage_selector.h. Performs feasibility checks, computes delays,
//   evaluates perfect hiding given fixed recompute ratio, computes cost over time,
//   and ranks devices based on optimization preference.
//
// Parts:
//   Part 1: Helpers (KV size computation).
//   Part 2: Feasibility & cost utilities.
//   Part 3: Selection logic and ranking.

#include "storage_selector.h"
#include <algorithm>
#include <iostream>

/////////////////////////////
// Part 1: Internal Helpers //
/////////////////////////////

static double ComputeKVCacheSizeGB(const ModelConfig& model, int context_length) {
    double element_size;
    switch (model.quant_level) {
        case QuantizationLevel::FP16: element_size = 2.0; break;
        case QuantizationLevel::INT8: element_size = 1.0; break;
        case QuantizationLevel::INT4: element_size = 0.5; break;
        default: element_size = 2.0; break;
    }
    double total_bytes = double(context_length) * model.num_layers * 2.0 * model.hidden_dim * element_size;
    double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
    return total_gb;
}

///////////////////////////////
// Part 2: Feasibility / Cost //
///////////////////////////////

bool StorageSelector::IsFeasible(const ModelConfig& model, int context_length, const StorageDevice& device) const {
    // Placeholder: insert capacity/access checks here
    return true;
}

double StorageSelector::ComputeStorageCostGBHour(double size_gb, double per_gb_hour) const {
    return size_gb * per_gb_hour;
}

/////////////////////////////////////
// Part 3: Selection & Ranking      //
/////////////////////////////////////

StorageSelectorOutput StorageSelector::SelectBestDevice(
    const ModelConfig& model,
    int context_length,
    const std::vector<std::shared_ptr<StorageDevice>>& candidates,
    const PrefillProfile& profile,
    const std::map<std::string, double>& device_cost_per_gb_hour,
    const std::optional<double>& full_prefill_time_ms_opt
) {
    std::vector<DevicePerformance> feasible_list;

    if (!full_prefill_time_ms_opt.has_value()) {
        std::cerr << "[StorageSelector] Warning: missing full prefill time; recompute delay fallback may be imprecise.\n";
    }
    double full_prefill_time_ms = full_prefill_time_ms_opt.value_or(1.0);

    for (const auto& dev_ptr : candidates) {
        const StorageDevice& dev = *dev_ptr;
        if (!IsFeasible(model, context_length, dev)) continue;

        // Compute loading & recompute delays
        LoadingController loader;
        double load_delay = loader.EstimateLoadingDelayMs(model, context_length, dev);
        double recompute_delay = loader.EstimateRecomputeDelayMs(full_prefill_time_ms, recompute_ratio_);
        bool hiding = recompute_delay <= load_delay;
        double total_delay = std::max(load_delay, recompute_delay);
        double kv_size_gb = ComputeKVCacheSizeGB(model, context_length);
        double per_gb_hour = 0.0;
        auto it_cost = device_cost_per_gb_hour.find(dev.name);
        if (it_cost != device_cost_per_gb_hour.end()) per_gb_hour = it_cost->second;

        double total_cost = 0.0;
        if (per_gb_hour > 0.0) {
            total_cost = ComputeStorageCostGBHour(kv_size_gb, per_gb_hour) * duration_hours_;
        }

        DevicePerformance perf{
            dev_ptr,
            load_delay,
            recompute_delay,
            hiding,
            total_delay,
            total_cost
        };
        feasible_list.push_back(perf);
    }

    // Ranking according to preference
    std::vector<DevicePerformance> filtered;
    if (preference_ == OptimizationPreference::COST) {
        for (auto& p : feasible_list) {
            if (p.perfect_hiding) filtered.push_back(p);
        }
        std::sort(filtered.begin(), filtered.end(), [](const DevicePerformance& a, const DevicePerformance& b) {
            if (a.total_storage_cost != b.total_storage_cost)
                return a.total_storage_cost < b.total_storage_cost;
            return a.total_delay_ms < b.total_delay_ms;
        });
    } else { // LATENCY
        filtered = feasible_list;
        std::sort(filtered.begin(), filtered.end(), [](const DevicePerformance& a, const DevicePerformance& b) {
            return a.total_delay_ms < b.total_delay_ms;
        });
    }

    StorageSelectorOutput out;
    if (!filtered.empty()) {
        out.selected_performance = filtered[0];
        out.selected_device = filtered[0].device;
        for (size_t i = 1; i < filtered.size(); ++i) {
            out.alternatives.push_back(filtered[i]);
        }
    } else if (!feasible_list.empty()) {
        // Fallback if strict filtering dropped all candidates
        std::sort(feasible_list.begin(), feasible_list.end(), [&](const DevicePerformance& a, const DevicePerformance& b) {
            if (preference_ == OptimizationPreference::COST) {
                if (a.total_storage_cost != b.total_storage_cost)
                    return a.total_storage_cost < b.total_storage_cost;
                return a.total_delay_ms < b.total_delay_ms;
            } else {
                return a.total_delay_ms < b.total_delay_ms;
            }
        });
        out.selected_performance = feasible_list[0];
        out.selected_device = feasible_list[0].device;
        for (size_t i = 1; i < feasible_list.size(); ++i) {
            out.alternatives.push_back(feasible_list[i]);
        }
    } else {
        std::cerr << "[StorageSelector] No feasible devices found.\n";
    }
    return out;
}
