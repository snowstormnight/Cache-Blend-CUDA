// test_loading_controller.cpp
//
// Unit test for LoadingController logic.
// Scenario: extremely fast storage such that loading delay is tiny, forcing the
// optimal recompute ratio to hit the minimum quality floor.

#include "loading_controller.h"
#include "test_utils.h"
#include <iostream>

int main() {
    std::cout << "=== test_loading_controller ===\n";

    // Model: small but arbitrary
    ModelConfig model{
        .name = "TestModel",
        .num_layers = 4,
        .hidden_dim = 64,
        .num_heads = 8,
        .quant_level = QuantizationLevel::FP16
    };
    int context_length = 100;

    // Prefill profile: assume full prefill takes 1000ms
    PrefillProfile profile;
    profile.full_prefill_time_ms[{model.name, context_length}] = 1000.0;

    // Storage: extremely high bandwidth to make load delay very small
    StorageDevice fast_storage{
        .name = "UltraFast",
        .bandwidth_gbps = 1e6, // astronomical to force transfer negligible
        .latency_ms = 0.5,
        .per_layer_overhead_ms = 0.1
    };

    LoadingController loader(0.15, 0.05); // min quality ratio 15%, 5% overhead
    auto out = loader.ComputeOptimalRatio(model, context_length, fast_storage, profile);

    // Expected: recompute delay constraint is very tight (load delay ~0.5 + 4*0.1 = 0.9ms)
    // So crossover ratio would be ~0.9 / (1000 * 1.05) ~ 0.000857 < min floor -> floor applied.
    expect_close("optimal recompute ratio", out.optimal_recompute_ratio, 0.15, 1e-4);
    expect_true("achieves hiding? (should be false or true depending on fallback)", out.achieves_hiding == (out.optimal_recompute_ratio * 1000.0 * 1.05 <= loader.EstimateLoadingDelayMs(model, context_length, fast_storage)));
    expect_close("quality score floor", out.quality_score, 0.8, 1e-6);
    std::cout << "Computed expected total delay (ms): " << out.expected_total_delay_ms << "\n";

    return 0;
}
