// storage_selector.h
//
// Overview:
//   Implements Part 2 of the controller: selecting the best storage device given a fixed recompute ratio.
//   Evaluates candidates by feasibility, computes their loading/recompute delays, checks perfect hiding,
//   calculates storage cost over duration, and ranks them based on optimization preference (cost or latency).
//
// Parts:
//   Part 1: Enums and result/data structures.
//   Part 2: StorageSelector class interface.

#pragma once
#include <string>
#include <vector>
#include <optional>
#include <map>
#include <memory>
#include "loading_controller.h"

///////////////////////
// Part 1: Data Types //
///////////////////////

enum class OptimizationPreference {
    COST,
    LATENCY
};

struct DevicePerformance {
    std::shared_ptr<StorageDevice> device;
    double loading_delay_ms;
    double recompute_delay_ms;
    bool perfect_hiding;
    double total_delay_ms;
    double total_storage_cost; // over configured duration
};

struct StorageSelectorOutput {
    std::shared_ptr<StorageDevice> selected_device;
    DevicePerformance selected_performance;
    std::vector<DevicePerformance> alternatives; // ranked list
};

/////////////////////////////
// Part 2: Selector Class   //
/////////////////////////////

class StorageSelector {
public:
    StorageSelector(OptimizationPreference opt_pref,
                    double storage_duration_hours,
                    double fixed_recompute_ratio)
        : preference_(opt_pref),
          duration_hours_(storage_duration_hours),
          recompute_ratio_(fixed_recompute_ratio) {}

    // Main API: provide model/context, candidates, prefill profile and cost map
    StorageSelectorOutput SelectBestDevice(
        const ModelConfig& model,
        int context_length,
        const std::vector<std::shared_ptr<StorageDevice>>& candidates,
        const PrefillProfile& profile,
        const std::map<std::string, double>& device_cost_per_gb_hour, // name -> cost per GB-hour
        const std::optional<double>& full_prefill_time_ms_opt // for recompute delay calculation
    );

private:
    bool IsFeasible(const ModelConfig& model, int context_length, const StorageDevice& device) const;
    double ComputeStorageCostGBHour(double size_gb, double per_gb_hour) const;

    OptimizationPreference preference_;
    double duration_hours_;
    double recompute_ratio_;
};
