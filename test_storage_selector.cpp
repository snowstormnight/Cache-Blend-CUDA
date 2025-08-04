// test_storage_selector.cpp
//
// Unit test for StorageSelector logic.
// Two devices: both can hide recompute at fixed ratio; costs differ.
// Verifies selection under COST and LATENCY preferences.

#include "storage_selector.h"
#include "test_utils.h"
#include <iostream>

int main() {
    std::cout << "=== test_storage_selector ===\n";

    // Simplified model to keep numbers manageable.
    ModelConfig model{
        .name = "SmallModel",
        .num_layers = 2,
        .hidden_dim = 16,
        .num_heads = 4,
        .quant_level = QuantizationLevel::INT8
    };
    int context_length = 64;

    // Prefill time for full prefill
    PrefillProfile profile;
    profile.full_prefill_time_ms[{model.name, context_length}] = 1000.0; // ms

    // Device A: decent hiding, higher cost
    auto devA = std::make_shared<StorageDevice>(StorageDevice{
        .name = "DeviceA",
        .bandwidth_gbps = 0.1, // makes transfer small but not crazy; we rely on overhead
        .latency_ms = 10.0,
        .per_layer_overhead_ms = 80.0 // 2 layers -> 160ms overhead
    });

    // Device B: slower but still hiding, cheaper
    auto devB = std::make_shared<StorageDevice>(StorageDevice{
        .name = "DeviceB",
        .bandwidth_gbps = 0.05,
        .latency_ms = 20.0,
        .per_layer_overhead_ms = 300.0 // 600ms overhead
    });

    std::vector<std::shared_ptr<StorageDevice>> candidates{devA, devB};

    // Cost per GB-hour: A is more expensive than B
    std::map<std::string, double> cost_map{
        {"DeviceA", 0.05},
        {"DeviceB", 0.02}
    };

    double fixed_ratio = 0.15; // fixed recompute ratio
    double duration_hours = 5.0;

    // === Test COST optimization: should pick the cheapest among those that achieve hiding ===
    StorageSelector selector_cost(OptimizationPreference::COST, duration_hours, fixed_ratio);
    auto out_cost = selector_cost.SelectBestDevice(model, context_length, candidates, profile, cost_map,
                                                  profile.get_prefill_time(model.name, context_length));
    expect_eq_str("COST preference selected device", out_cost.selected_device->name, "DeviceB");

    // === Test LATENCY optimization: should pick one with smaller total delay (DeviceA has ~170ms vs DeviceB ~620ms) ===
    StorageSelector selector_latency(OptimizationPreference::LATENCY, duration_hours, fixed_ratio);
    auto out_latency = selector_latency.SelectBestDevice(model, context_length, candidates, profile, cost_map,
                                                        profile.get_prefill_time(model.name, context_length));
    expect_eq_str("LATENCY preference selected device", out_latency.selected_device->name, "DeviceA");

    // Verify both picked devices satisfy hiding (recompute_delay <= load_delay)
    expect_true("DeviceB hiding (cost path)", out_cost.selected_performance.perfect_hiding);
    expect_true("DeviceA hiding (latency path)", out_latency.selected_performance.perfect_hiding);

    std::cout << "DeviceA total delay (ms): " << out_latency.selected_performance.total_delay_ms << "\n";
    std::cout << "DeviceB total delay (ms): " << out_cost.selected_performance.total_delay_ms << "\n";
    std::cout << "DeviceA cost over duration: " << out_latency.selected_performance.total_storage_cost << "\n";
    std::cout << "DeviceB cost over duration: " << out_cost.selected_performance.total_storage_cost << "\n";

    return 0;
}
