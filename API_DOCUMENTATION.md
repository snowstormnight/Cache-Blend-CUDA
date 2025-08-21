# Comprehensive API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [LoadingController API](#loadingcontroller-api)
4. [StorageSelector API](#storageselector-api)
5. [Data Structures](#data-structures)
6. [CUDA Acceleration APIs](#cuda-acceleration-apis)
7. [Test Utilities](#test-utilities)
8. [Integration Examples](#integration-examples)
9. [Build Instructions](#build-instructions)

## Overview

This project implements an intelligent KV cache management system for Large Language Models (LLMs) that optimizes the trade-off between storage loading and selective token recomputation. The system consists of two main components:

1. **LoadingController**: Optimizes the recompute ratio for a given storage device
2. **StorageSelector**: Selects the best storage device for a fixed recompute ratio

The system aims to minimize time-to-first-token (TTFT) while maintaining generation quality through strategic partial recomputation of tokens.

## Core Components

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                      │
│                      (bond.cpp)                          │
└─────────────────┬───────────────────────┬────────────────┘
                  │                       │
┌─────────────────▼──────────┐ ┌─────────▼──────────────┐
│    LoadingController       │ │    StorageSelector     │
│  (loading_controller.h/cpp)│ │ (storage_selector.h/cpp)│
└─────────────────┬──────────┘ └─────────┬──────────────┘
                  │                       │
┌─────────────────▼───────────────────────▼────────────────┐
│              Shared Data Structures                       │
│        (ModelConfig, StorageDevice, PrefillProfile)       │
└───────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│                   CUDA Acceleration                        │
│                    (fusor_cache.cu)                       │
└───────────────────────────────────────────────────────────┘
```

## LoadingController API

### Class: `LoadingController`

The LoadingController optimizes the recompute ratio for selective token recomputation given a fixed storage device.

#### Constructor
```cpp
LoadingController(double min_quality_ratio = 0.15, double overhead_factor = 0.05)
```

**Parameters:**
- `min_quality_ratio`: Minimum recompute ratio to maintain generation quality (default: 0.15)
- `overhead_factor`: Additional overhead for selective recompute mechanism (default: 0.05)

**Example:**
```cpp
LoadingController controller(0.15, 0.05);  // 15% min quality, 5% overhead
```

#### Primary Method: `ComputeOptimalRatio`
```cpp
LoadingControllerOutput ComputeOptimalRatio(
    const ModelConfig& model,
    int context_length,
    const StorageDevice& storage,
    const PrefillProfile& profile
)
```

**Purpose:** Computes the optimal recompute ratio that maximizes efficiency while maintaining quality.

**Parameters:**
- `model`: Configuration of the LLM model
- `context_length`: Number of tokens in the context
- `storage`: Storage device specifications
- `profile`: Prefill timing profile data

**Returns:** `LoadingControllerOutput` structure containing:
- `optimal_recompute_ratio`: Percentage of tokens to recompute [0,1]
- `achieves_hiding`: Whether recomputation is hidden by loading time
- `expected_total_delay_ms`: Total delay in milliseconds
- `quality_score`: Predicted quality metric

**Example:**
```cpp
ModelConfig model{
    .name = "Mistral-7B",
    .num_layers = 32,
    .hidden_dim = 4096,
    .num_heads = 32,
    .quant_level = QuantizationLevel::FP16
};

StorageDevice storage{
    .name = "NVMe SSD",
    .bandwidth_gbps = 20.0,
    .latency_ms = 0.5,
    .per_layer_overhead_ms = 0.1
};

PrefillProfile profile;
profile.full_prefill_time_ms[{"Mistral-7B", 2048}] = 1200.0;

LoadingController controller;
auto result = controller.ComputeOptimalRatio(model, 2048, storage, profile);

std::cout << "Optimal ratio: " << result.optimal_recompute_ratio * 100 << "%\n";
std::cout << "Perfect hiding: " << (result.achieves_hiding ? "Yes" : "No") << "\n";
```

#### Helper Methods

##### `EstimateLoadingDelayMs`
```cpp
double EstimateLoadingDelayMs(
    const ModelConfig& model, 
    int context_length, 
    const StorageDevice& storage
) const
```

**Purpose:** Estimates the time to load KV cache from storage to GPU memory.

**Returns:** Loading delay in milliseconds.

**Example:**
```cpp
double loading_time = controller.EstimateLoadingDelayMs(model, 2048, storage);
std::cout << "Loading delay: " << loading_time << " ms\n";
```

##### `EstimateRecomputeDelayMs`
```cpp
double EstimateRecomputeDelayMs(double full_prefill_time_ms, double ratio) const
```

**Purpose:** Estimates time to selectively recompute a percentage of tokens.

**Parameters:**
- `full_prefill_time_ms`: Time for full prefill
- `ratio`: Recompute ratio [0,1]

**Returns:** Recompute delay in milliseconds.

**Example:**
```cpp
double recompute_time = controller.EstimateRecomputeDelayMs(1200.0, 0.15);
std::cout << "Recompute delay for 15%: " << recompute_time << " ms\n";
```

##### `PredictQualityScore`
```cpp
double PredictQualityScore(double ratio) const
```

**Purpose:** Predicts generation quality based on recompute ratio.

**Returns:** Quality score [0,1].

**Example:**
```cpp
double quality = controller.PredictQualityScore(0.15);
std::cout << "Predicted quality at 15% recompute: " << quality << "\n";
```

## StorageSelector API

### Class: `StorageSelector`

The StorageSelector chooses the optimal storage device from candidates given a fixed recompute ratio.

#### Constructor
```cpp
StorageSelector(
    OptimizationPreference opt_pref,
    double storage_duration_hours,
    double fixed_recompute_ratio
)
```

**Parameters:**
- `opt_pref`: Optimization preference (COST or LATENCY)
- `storage_duration_hours`: Expected storage duration
- `fixed_recompute_ratio`: Fixed recompute ratio [0,1]

**Example:**
```cpp
StorageSelector selector(
    OptimizationPreference::COST,  // Optimize for cost
    24.0,                          // 24 hours storage
    0.15                           // 15% recompute ratio
);
```

#### Primary Method: `SelectBestDevice`
```cpp
StorageSelectorOutput SelectBestDevice(
    const ModelConfig& model,
    int context_length,
    const std::vector<std::shared_ptr<StorageDevice>>& candidates,
    const PrefillProfile& profile,
    const std::map<std::string, double>& device_cost_per_gb_hour,
    const std::optional<double>& full_prefill_time_ms_opt
)
```

**Purpose:** Selects the best storage device based on optimization criteria.

**Parameters:**
- `model`: LLM model configuration
- `context_length`: Number of tokens
- `candidates`: List of available storage devices
- `profile`: Prefill timing profile
- `device_cost_per_gb_hour`: Cost map (device name → cost per GB-hour)
- `full_prefill_time_ms_opt`: Optional prefill time for recompute calculation

**Returns:** `StorageSelectorOutput` containing:
- `selected_device`: Best storage device
- `selected_performance`: Performance metrics
- `alternatives`: Ranked list of alternatives

**Example:**
```cpp
// Create storage device candidates
auto nvme = std::make_shared<StorageDevice>(StorageDevice{
    .name = "NVMe Local",
    .bandwidth_gbps = 20.0,
    .latency_ms = 0.5,
    .per_layer_overhead_ms = 0.1
});

auto remote_ssd = std::make_shared<StorageDevice>(StorageDevice{
    .name = "Remote SSD",
    .bandwidth_gbps = 8.0,
    .latency_ms = 2.0,
    .per_layer_overhead_ms = 0.2
});

std::vector<std::shared_ptr<StorageDevice>> candidates{nvme, remote_ssd};

// Define storage costs
std::map<std::string, double> cost_map{
    {"NVMe Local", 0.0},      // Local storage (no cost)
    {"Remote SSD", 0.02}      // $0.02 per GB-hour
};

// Select best device
StorageSelector selector(OptimizationPreference::COST, 10.0, 0.15);
auto result = selector.SelectBestDevice(
    model, 
    2048, 
    candidates, 
    profile, 
    cost_map,
    1200.0  // Full prefill time
);

if (result.selected_device) {
    std::cout << "Selected: " << result.selected_device->name << "\n";
    std::cout << "Total delay: " << result.selected_performance.total_delay_ms << " ms\n";
    std::cout << "Storage cost: $" << result.selected_performance.total_storage_cost << "\n";
}
```

## Data Structures

### Enum: `QuantizationLevel`
```cpp
enum class QuantizationLevel {
    FP16,   // 16-bit floating point
    INT8,   // 8-bit integer
    INT4    // 4-bit integer
};
```

**Usage:** Specifies model weight quantization level.

### Struct: `ModelConfig`
```cpp
struct ModelConfig {
    std::string name;           // Model identifier
    int num_layers;             // Number of transformer layers
    int hidden_dim;             // Hidden dimension size
    int num_heads;              // Number of attention heads
    QuantizationLevel quant_level;  // Quantization level
};
```

**Example:**
```cpp
ModelConfig llama_config{
    .name = "Llama-70B",
    .num_layers = 80,
    .hidden_dim = 8192,
    .num_heads = 64,
    .quant_level = QuantizationLevel::INT8
};
```

### Struct: `StorageDevice`
```cpp
struct StorageDevice {
    std::string name;              // Device identifier
    double bandwidth_gbps;         // Bandwidth in GB/s
    double latency_ms;             // Access latency in milliseconds
    double per_layer_overhead_ms;  // Per-layer coordination overhead
};
```

**Example:**
```cpp
StorageDevice gpu_hbm{
    .name = "GPU HBM",
    .bandwidth_gbps = 900.0,
    .latency_ms = 0.001,
    .per_layer_overhead_ms = 0.0
};
```

### Struct: `PrefillProfile`
```cpp
struct PrefillProfile {
    std::map<std::pair<std::string, int>, double> full_prefill_time_ms;
    
    std::optional<double> get_prefill_time(
        const std::string& model_name, 
        int context_length
    ) const;
};
```

**Purpose:** Stores profiled prefill timing data for different model-context combinations.

**Example:**
```cpp
PrefillProfile profile;
// Add profiling data
profile.full_prefill_time_ms[{"Mistral-7B", 1024}] = 600.0;
profile.full_prefill_time_ms[{"Mistral-7B", 2048}] = 1200.0;
profile.full_prefill_time_ms[{"Llama-70B", 2048}] = 8500.0;

// Query profiling data
auto time = profile.get_prefill_time("Mistral-7B", 2048);
if (time.has_value()) {
    std::cout << "Prefill time: " << time.value() << " ms\n";
}
```

### Struct: `LoadingControllerOutput`
```cpp
struct LoadingControllerOutput {
    double optimal_recompute_ratio;  // [0,1]
    bool achieves_hiding;            // Perfect hiding achieved
    double expected_total_delay_ms;  // Total delay
    double quality_score;            // Predicted quality [0,1]
};
```

### Enum: `OptimizationPreference`
```cpp
enum class OptimizationPreference {
    COST,     // Minimize storage cost
    LATENCY   // Minimize total latency
};
```

### Struct: `DevicePerformance`
```cpp
struct DevicePerformance {
    std::shared_ptr<StorageDevice> device;
    double loading_delay_ms;      // Loading time
    double recompute_delay_ms;     // Recompute time
    bool perfect_hiding;           // Recompute hidden by loading
    double total_delay_ms;         // Max(loading, recompute)
    double total_storage_cost;     // Cost over duration
};
```

### Struct: `StorageSelectorOutput`
```cpp
struct StorageSelectorOutput {
    std::shared_ptr<StorageDevice> selected_device;
    DevicePerformance selected_performance;
    std::vector<DevicePerformance> alternatives;  // Ranked alternatives
};
```

## CUDA Acceleration APIs

### File: `fusor_cache.cu`

The CUDA implementation provides GPU-accelerated KV cache fusion with selective recomputation.

#### Main Function: `run_fusor`
```cpp
void run_fusor(const int num_layers, const int tokens, const float recompute_ratio)
```

**Purpose:** Performs layer-parallel KV cache loading and selective recomputation on GPU.

**Parameters:**
- `num_layers`: Number of transformer layers
- `tokens`: Number of tokens to process
- `recompute_ratio`: Percentage of tokens to recompute

**Implementation Details:**
- Uses CUDA streams for layer-parallel execution
- Overlaps cache loading with recomputation
- Implements HKVD (High KV Deviation) selection for critical tokens

**Example Usage:**
```bash
# Compile CUDA code
nvcc -o fusor_cache fusor_cache.cu

# Run with custom parameters
./fusor_cache 0.15 32 2048  # 15% recompute, 32 layers, 2048 tokens
```

#### CUDA Kernel: `recomputeKV`
```cpp
__global__ void recomputeKV(
    float* kv, 
    const int* hkvd_flags, 
    int num_tokens, 
    int dim
)
```

**Purpose:** GPU kernel for selective token recomputation based on HKVD flags.

**Parameters:**
- `kv`: KV cache data (tokens × hidden_dim)
- `hkvd_flags`: Binary flags indicating tokens to recompute
- `num_tokens`: Total number of tokens
- `dim`: Hidden dimension size

#### Helper Functions

##### `simulate_cache_load`
```cpp
void simulate_cache_load(
    float*& d_kv, 
    int num_tokens, 
    int dim, 
    cudaStream_t stream
)
```

**Purpose:** Simulates asynchronous KV cache loading from storage to GPU.

##### `select_HKVD`
```cpp
void select_HKVD(std::vector<int>& flags, float ratio)
```

**Purpose:** Selects top-k tokens for recomputation based on deviation scores.

### Constants
```cpp
#define MAX_LAYERS 32      // Maximum supported layers
#define MAX_TOKENS 2048    // Maximum token count
#define HIDDEN_DIM 64      // Hidden dimension for testing
#define MAX_CHUNKS 10      // Maximum cache chunks
```

## Test Utilities

### File: `test_utils.h`

Provides lightweight testing utilities without external dependencies.

#### Function: `expect_close`
```cpp
void expect_close(const char* name, double a, double b, double tol = 1e-3)
```

**Purpose:** Asserts two floating-point values are approximately equal.

**Example:**
```cpp
double computed = 1200.5;
double expected = 1200.0;
expect_close("Loading delay", computed, expected, 1.0);  // tolerance = 1.0
// Output: [PASS] Loading delay: 1200.5 ≈ 1200
```

#### Function: `expect_true`
```cpp
void expect_true(const char* name, bool cond)
```

**Purpose:** Asserts a boolean condition is true.

**Example:**
```cpp
bool hiding_achieved = true;
expect_true("Perfect hiding achieved", hiding_achieved);
// Output: [PASS] Perfect hiding achieved
```

#### Function: `expect_eq_str`
```cpp
void expect_eq_str(const char* name, const std::string& a, const std::string& b)
```

**Purpose:** Asserts two strings are equal.

**Example:**
```cpp
std::string device = "NVMe SSD";
expect_eq_str("Selected device", device, "NVMe SSD");
// Output: [PASS] Selected device: "NVMe SSD" == "NVMe SSD"
```

## Integration Examples

### Example 1: Basic Usage
```cpp
#include "loading_controller.h"
#include "storage_selector.h"

int main() {
    // Step 1: Configure model
    ModelConfig model{
        .name = "Mistral-7B",
        .num_layers = 32,
        .hidden_dim = 4096,
        .num_heads = 32,
        .quant_level = QuantizationLevel::FP16
    };
    
    // Step 2: Set up profiling data
    PrefillProfile profile;
    profile.full_prefill_time_ms[{model.name, 2048}] = 1200.0;
    
    // Step 3: Find optimal recompute ratio
    StorageDevice storage{
        .name = "NVMe SSD",
        .bandwidth_gbps = 20.0,
        .latency_ms = 0.5,
        .per_layer_overhead_ms = 0.1
    };
    
    LoadingController controller;
    auto optimal = controller.ComputeOptimalRatio(
        model, 2048, storage, profile
    );
    
    std::cout << "Optimal ratio: " << optimal.optimal_recompute_ratio << "\n";
    
    return 0;
}
```

### Example 2: Multi-Device Selection
```cpp
#include "storage_selector.h"

int main() {
    // Create multiple storage options
    auto devices = std::vector<std::shared_ptr<StorageDevice>>{
        std::make_shared<StorageDevice>(StorageDevice{
            "GPU HBM", 900.0, 0.001, 0.0
        }),
        std::make_shared<StorageDevice>(StorageDevice{
            "CPU RAM", 100.0, 0.01, 0.05
        }),
        std::make_shared<StorageDevice>(StorageDevice{
            "NVMe SSD", 20.0, 0.5, 0.1
        })
    };
    
    // Define costs
    std::map<std::string, double> costs{
        {"GPU HBM", 0.0},
        {"CPU RAM", 0.0},
        {"NVMe SSD", 0.01}
    };
    
    // Select best device for cost optimization
    StorageSelector selector(
        OptimizationPreference::COST,
        24.0,  // 24 hours
        0.15   // 15% recompute
    );
    
    auto result = selector.SelectBestDevice(
        model, 2048, devices, profile, costs, 1200.0
    );
    
    // Display results
    if (result.selected_device) {
        std::cout << "Best device: " << result.selected_device->name << "\n";
        
        for (const auto& alt : result.alternatives) {
            std::cout << "Alternative: " << alt.device->name 
                     << " (delay: " << alt.total_delay_ms << " ms)\n";
        }
    }
    
    return 0;
}
```

### Example 3: Dynamic Adaptation
```cpp
class AdaptiveController {
private:
    LoadingController loader_;
    ModelConfig model_;
    PrefillProfile profile_;
    
public:
    void adapt_to_conditions(
        const StorageDevice& current_device,
        int context_length
    ) {
        // Monitor and adapt
        auto result = loader_.ComputeOptimalRatio(
            model_, context_length, current_device, profile_
        );
        
        if (!result.achieves_hiding) {
            std::cout << "Warning: Cannot hide recomputation\n";
            std::cout << "Consider switching to faster storage\n";
            
            // Trigger device reselection
            trigger_device_reselection(context_length);
        }
        
        if (result.quality_score < 0.9) {
            std::cout << "Quality degradation detected\n";
            std::cout << "Reducing recompute ratio\n";
        }
    }
    
    void trigger_device_reselection(int context_length) {
        // Implementation for switching storage devices
        // ...
    }
};
```

### Example 4: Testing Implementation
```cpp
#include "test_utils.h"
#include "loading_controller.h"

void test_loading_controller() {
    std::cout << "=== Testing LoadingController ===\n";
    
    ModelConfig model{
        .name = "test-model",
        .num_layers = 10,
        .hidden_dim = 1024,
        .num_heads = 16,
        .quant_level = QuantizationLevel::FP16
    };
    
    StorageDevice storage{
        .name = "test-storage",
        .bandwidth_gbps = 10.0,
        .latency_ms = 1.0,
        .per_layer_overhead_ms = 0.1
    };
    
    PrefillProfile profile;
    profile.full_prefill_time_ms[{"test-model", 512}] = 300.0;
    
    LoadingController controller(0.15, 0.05);
    
    // Test loading delay estimation
    double loading_delay = controller.EstimateLoadingDelayMs(
        model, 512, storage
    );
    expect_close("Loading delay", loading_delay, 42.0, 5.0);
    
    // Test recompute delay
    double recompute_delay = controller.EstimateRecomputeDelayMs(
        300.0, 0.15
    );
    expect_close("Recompute delay", recompute_delay, 47.25, 1.0);
    
    // Test quality prediction
    double quality = controller.PredictQualityScore(0.15);
    expect_close("Quality score", quality, 0.95, 0.05);
    
    // Test optimal ratio computation
    auto result = controller.ComputeOptimalRatio(
        model, 512, storage, profile
    );
    expect_true("Valid ratio", result.optimal_recompute_ratio >= 0.15);
    expect_true("Achieves hiding", result.achieves_hiding);
}

int main() {
    test_loading_controller();
    return 0;
}
```

## Build Instructions

### CMake Build
```cmake
cmake_minimum_required(VERSION 3.10)
project(KVCacheController)

set(CMAKE_CXX_STANDARD 17)

# Main library
add_library(kv_controller
    loading_controller.cpp
    storage_selector.cpp
)

# Main executable
add_executable(bond bond.cpp)
target_link_libraries(bond kv_controller)

# Tests
add_executable(test_loading test_loading_controller.cpp)
target_link_libraries(test_loading kv_controller)

add_executable(test_storage test_storage_selector.cpp)
target_link_libraries(test_storage kv_controller)

# CUDA support (optional)
find_package(CUDA)
if(CUDA_FOUND)
    cuda_add_executable(fusor_cache fusor_cache.cu)
endif()
```

### Manual Compilation
```bash
# Compile C++ components
g++ -std=c++17 -O2 -c loading_controller.cpp -o loading_controller.o
g++ -std=c++17 -O2 -c storage_selector.cpp -o storage_selector.o
g++ -std=c++17 -O2 bond.cpp loading_controller.o storage_selector.o -o bond

# Compile tests
g++ -std=c++17 test_loading_controller.cpp loading_controller.o -o test_loading
g++ -std=c++17 test_storage_selector.cpp loading_controller.o storage_selector.o -o test_storage

# Compile CUDA component (if CUDA is available)
nvcc -O2 fusor_cache.cu -o fusor_cache
```

### Running Tests
```bash
# Run unit tests
./test_loading
./test_storage

# Run integration example
./bond

# Run CUDA acceleration (if compiled)
./fusor_cache 0.15 32 2048
```

## Performance Considerations

### Memory Requirements
- KV cache size = `2 × num_layers × context_length × hidden_dim × element_size`
- Element sizes: FP16 (2 bytes), INT8 (1 byte), INT4 (0.5 bytes)

### Optimization Tips
1. **Profile First**: Always collect accurate prefill timing data for your specific hardware
2. **Balance Quality**: Start with 15% recompute ratio for quality preservation
3. **Layer Parallelism**: Use CUDA streams for concurrent layer processing
4. **Storage Selection**: Consider both bandwidth and latency when choosing storage
5. **Cost vs Performance**: Use appropriate OptimizationPreference based on deployment

### Benchmarking
```cpp
#include <chrono>

template<typename Func>
double benchmark(Func func, int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

// Usage
double avg_time = benchmark([&]() {
    controller.ComputeOptimalRatio(model, 2048, storage, profile);
});
std::cout << "Average computation time: " << avg_time << " ms\n";
```

## Troubleshooting

### Common Issues

1. **Missing Prefill Data**
   - Error: `get_prefill_time` returns `std::nullopt`
   - Solution: Ensure profile data includes the model-context combination

2. **Invalid Recompute Ratio**
   - Error: Ratio below minimum quality threshold
   - Solution: Adjust `min_quality_ratio` or use faster storage

3. **No Feasible Device**
   - Error: All devices filtered out
   - Solution: Check device capacity and bandwidth requirements

4. **CUDA Out of Memory**
   - Error: CUDA malloc fails
   - Solution: Reduce batch size or use smaller test dimensions

### Debug Output
```cpp
// Enable debug logging
#define DEBUG_MODE

#ifdef DEBUG_MODE
#define DEBUG_LOG(msg) std::cout << "[DEBUG] " << msg << std::endl
#else
#define DEBUG_LOG(msg)
#endif

// Usage in code
DEBUG_LOG("Computing optimal ratio for " << model.name);
DEBUG_LOG("Loading delay: " << loading_delay << " ms");
```

## License and Contributing

This documentation covers the public APIs and usage of the KV Cache Controller system. For contribution guidelines and license information, please refer to the project repository.

## Version History

- **v1.0.0**: Initial release with LoadingController and StorageSelector
- **v1.1.0**: Added CUDA acceleration support
- **v1.2.0**: Enhanced profiling and quality prediction

---

*Generated: API Documentation v1.2.0*
*Last Updated: 2024*