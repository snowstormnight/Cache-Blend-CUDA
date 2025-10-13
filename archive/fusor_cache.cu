#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define MAX_LAYERS 32
#define MAX_TOKENS 2048
#define HIDDEN_DIM 64
#define MAX_CHUNKS 10

struct KVCache {
    float* data;  // Shape: [tokens x hidden_dim]
    int tokens;
};

__global__ void recomputeKV(float* kv, const int* hkvd_flags, int num_tokens, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_tokens && hkvd_flags[i]) {
        for (int d = 0; d < dim; ++d) {
            kv[i * dim + d] += 1.0f;
        }
    }
}

// Random fill for simulating chunk KV load
void fill_random_kv(float* host_ptr, int num_tokens, int dim) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < num_tokens * dim; ++i) {
        host_ptr[i] = dist(rng);
    }
}

// Simulates fetching the precomputed KV cache from "SSD/RAM"
void simulate_cache_load(float*& d_kv, int num_tokens, int dim, cudaStream_t stream) {
    float* host_kv = new float[num_tokens * dim];
    fill_random_kv(host_kv, num_tokens, dim);
    cudaMallocAsync(&d_kv, sizeof(float) * num_tokens * dim, stream);
    cudaMemcpyAsync(d_kv, host_kv, sizeof(float) * num_tokens * dim, cudaMemcpyHostToDevice, stream);
    delete[] host_kv;
}

// Select top-k tokens with fake deviation scores
void select_HKVD(std::vector<int>& flags, float ratio) {
    int topk = static_cast<int>(flags.size() * ratio);
    for (int i = 0; i < topk; ++i) flags[i] = 1;
}

// MAIN FUSOR ENTRY
void run_fusor(const int num_layers, const int tokens, const float recompute_ratio) {
    const int kv_dim = HIDDEN_DIM;
    const size_t kv_bytes = tokens * kv_dim * sizeof(float);

    std::vector<cudaStream_t> streams(num_layers);
    std::vector<float*> d_kv_fused(num_layers);
    std::vector<int*> d_hkvd_flags(num_layers);

    auto start_total = std::chrono::high_resolution_clock::now();

    for (int layer = 0; layer < num_layers; ++layer) {
        cudaStreamCreate(&streams[layer]);

        // Allocate and load precomputed KV cache
        simulate_cache_load(d_kv_fused[layer], tokens, kv_dim, streams[layer]);

        // Generate HKVD mask
        std::vector<int> h_flags(tokens, 0);
        select_HKVD(h_flags, recompute_ratio);
        cudaMallocAsync(&d_hkvd_flags[layer], sizeof(int) * tokens, streams[layer]);
        cudaMemcpyAsync(d_hkvd_flags[layer], h_flags.data(), sizeof(int) * tokens, cudaMemcpyHostToDevice, streams[layer]);

        // Launch recompute kernel (overlap with next layer's load)
        int threads = 256;
        int blocks = (tokens + threads - 1) / threads;
        recomputeKV<<<blocks, threads, 0, streams[layer]>>>(d_kv_fused[layer], d_hkvd_flags[layer], tokens, kv_dim);
    }

    // Sync and cleanup
    for (int layer = 0; layer < num_layers; ++layer) {
        cudaStreamSynchronize(streams[layer]);
        cudaFreeAsync(d_kv_fused[layer], streams[layer]);
        cudaFreeAsync(d_hkvd_flags[layer], streams[layer]);
        cudaStreamDestroy(streams[layer]);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end_total - start_total).count();

    std::cout << "[Fusor] Fusion complete across " << num_layers << " layers. Time = " << ms << " ms\n";
}

int main(int argc, char** argv) {
    int layers = 10;
    int tokens = 1024;
    float recompute_ratio = 0.15f;

    if (argc > 1) recompute_ratio = std::stof(argv[1]);
    if (argc > 2) layers = std::stoi(argv[2]);
    if (argc > 3) tokens = std::stoi(argv[3]);

    std::cout << "Launching Fusor with " << layers << " layers, " << tokens << " tokens, recompute_ratio = " << recompute_ratio << "\n";
    run_fusor(layers, tokens, recompute_ratio);
    cudaDeviceReset();
    return 0;
}
