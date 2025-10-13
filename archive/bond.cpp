// bond.cpp
//
// Overview:
//   "Bond" file that ties Part 1 (LoadingController) and Part 2 (StorageSelector) together.
//   Demonstrates typical usage: compute optimal recompute ratio for a chosen storage device,
//   then, with a fixed ratio, evaluate and select among multiple storage candidates.
//   Outputs diagnostics for review.
//
// Parts:
//   Part 0: Input setup (model, context, profile).
//   Part 1: Invoke LoadingController for optimal ratio on a chosen device.
//   Part 2: Invoke StorageSelector to pick best device under fixed recompute ratio.
//   Part 3: Reporting results and alternatives.

#include <iostream>
#include <memory>
#include <vector>
#include <iomanip>
#include "loading_controller.h"
#include "storage_selector.h"

int main() {
    // === Part 0: Setup inputs ===
    ModelConfig model{
        .name = "Mistral-7B",
        .num_layers = 32,
        .hidden_dim = 4096,
        .num_heads = 32,
        .quant_level = QuantizationLevel::FP16
    };
    int context_length = 2048;

    PrefillProfile profile;
    profile.full_prefill_time_ms[{model.name, context_length}] = 1200.0; // example full prefill time in ms

    // === Part 1: LoadingController (optimal recompute ratio) ===
    StorageDevice chosen_storage{
        .name = "NVMe Local",
        .bandwidth_gbps = 20.0,
        .latency_ms = 0.5,
        .per_layer_overhead_ms = 0.1
    };
    LoadingController loader(0.15, 0.05);
    auto load_out = loader.ComputeOptimalRatio(model, context_length, chosen_storage, profile);

    std::cout << "=== [Part 1] Loading Controller Output ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Optimal recompute ratio: " << load_out.optimal_recompute_ratio * 100 << "%\n";
    std::cout << "Achieves perfect hiding: " << (load_out.achieves_hiding ? "yes" : "no") << "\n";
    std::cout << "Expected total delay (ms): " << load_out.expected_total_delay_ms << "\n";
    std::cout << "Predicted quality score: " << load_out.quality_score << "\n\n";

    // === Part 2: StorageSelector (fixed recompute ratio) ===
    auto dev1 = std::make_shared<StorageDevice>(StorageDevice{
        .name = "NVMe Local",
        .bandwidth_gbps = 20.0,
        .latency_ms = 0.5,
        .per_layer_overhead_ms = 0.1
    });
    auto dev2 = std::make_shared<StorageDevice>(StorageDevice{
        .name = "Remote SSD",
        .bandwidth_gbps = 8.0,
        .latency_ms = 2.0,
        .per_layer_overhead_ms = 0.2
    });
    std::vector<std::shared_ptr<StorageDevice>> candidates{dev1, dev2};

    std::map<std::string, double> cost_map{
        {"NVMe Local", 0.0},
        {"Remote SSD", 0.02} // cost per GB-hour
    };

    double fixed_ratio = 0.15; // empirical quality-preserving ratio
    StorageSelector selector(OptimizationPreference::COST, /*duration_hours=*/10.0, fixed_ratio);
    auto select_out = selector.SelectBestDevice(
        model,
        context_length,
        candidates,
        profile,
        cost_map,
        profile.get_prefill_time(model.name, context_length)
    );

    // === Part 3: Reporting ===
    std::cout << "=== [Part 2] Storage Selector Output ===\n";
    if (select_out.selected_device) {
        std::cout << "Selected device: " << select_out.selected_device->name << "\n";
        auto& perf = select_out.selected_performance;
        std::cout << "Loading delay (ms): " << perf.loading_delay_ms << "\n";
        std::cout << "Recompute delay (ms): " << perf.recompute_delay_ms << "\n";
        std::cout << "Perfect hiding: " << (perf.perfect_hiding ? "yes" : "no") << "\n";
        std::cout << "Total delay (ms): " << perf.total_delay_ms << "\n";
        std::cout << "Total storage cost over duration: " << perf.total_storage_cost << "\n";
    } else {
        std::cout << "No device selected.\n";
    }

    if (!select_out.alternatives.empty()) {
        std::cout << "\nAlternatives:\n";
        for (const auto& alt : select_out.alternatives) {
            std::cout << " - " << alt.device->name
                      << " | total delay(ms): " << alt.total_delay_ms
                      << " | hiding: " << (alt.perfect_hiding ? "yes" : "no")
                      << " | cost over duration: " << alt.total_storage_cost << "\n";
        }
    }

    return 0;
}
