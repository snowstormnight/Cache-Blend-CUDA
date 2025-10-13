# Fusor + Cache Module – Two-Part Design

## Part 1: KV Cache Fusion (Selective Recompute)

### Description

This component merges multiple pre-computed KV caches from different text chunks into a single fused KV cache for the input sequence. To preserve generation quality, it selectively recomputes the KV values for tokens with the highest deviation while reusing the rest. The recomputation is performed layer-by-layer, pipelined with cache loading to hide recompute costs.

### Input Description

* **Model Configuration**: The LLM model’s architecture including the number of layers, hidden size, attention heads, and quantization level.
* **Retrieved KV Caches**: A set of pre-computed KV caches for the text chunks relevant to the current input.
* **Recompute Ratio**: Percentage of tokens to selectively recompute per layer, determined by the loading controller.
* **Context Composition**: The ordered sequence of text chunks plus user query tokens forming the full model input.
* **Device Configuration**: GPU properties such as memory bandwidth and compute throughput for recomputation.
* **KV Deviation Profile**: A per-layer deviation ranking for selecting high-KV-deviation tokens.
* **Prefill Profile Data**: Measured prefill performance for different context lengths and model configurations.

### Output Description

* **Fused KV Cache**: A KV cache that combines pre-computed caches and selectively recomputed HKVD tokens for all layers.
* **Recompute Cost**: The GPU time spent on selective KV recomputation.
* **Attention Quality Score**: An estimated metric of cross-chunk attention fidelity relative to full prefill.
* **Fusion Log**: Diagnostic data, including the percentage of recomputed tokens per layer and cache reuse statistics.

### Core Functions

#### Identify HKVD Tokens

Select tokens with the highest KV deviation by comparing pre-computed KV values with layer-wise heuristics or prior layer HKVD selections. This function ensures sparsity (typically less than or equal to 15% of tokens) to minimize recomputation cost.

#### Layer-Wise Fusion

For each transformer layer:

1. Load the pre-computed KV cache for that layer from storage.
2. Recompute the KV for HKVD tokens using GPU kernels.
3. Merge recomputed KV entries with the rest of the pre-computed cache.
4. Pass the fused KV cache to the next layer.

#### Pipelined Execution

Overlaps KV cache loading for layer *i+1* with recomputation for layer *i* using asynchronous GPU streams, ensuring recompute cost is hidden by loading whenever possible.

#### Attention Deviation Monitoring

Computes the attention deviation between fused KV and full prefill for quality assurance. If deviation exceeds thresholds, the recompute ratio can be dynamically increased.

---

## Part 2: KV Cache Store (Storage and Retrieval)

### Description

This component manages the persistent storage and retrieval of KV caches across requests. It splits inputs into chunks, hashes each chunk for efficient lookup, retrieves KV caches from the appropriate storage tier, and updates the cache store with newly generated KV caches.

### Input Description

* **Chunked Input Texts**: List of text chunks produced by the retrieval or chunking layer.
* **Storage Hierarchy**: A prioritized list of storage tiers (GPU memory, CPU RAM, SSD) along with their bandwidth, latency, and capacity.
* **KV Cache Hash Map**: A hash table mapping chunk hashes to stored KV cache locations.
* **Eviction Policy**: Policy for freeing storage (e.g., Least Recently Used).
* **Device Constraints**: Available GPU memory and background I/O bandwidth for cache transfers.

### Output Description

* **Fetched KV Caches**: KV caches for all reusable chunks loaded into GPU memory.
* **Cache Misses**: List of chunks for which KV caches were not found and must be computed.
* **Eviction Log**: Record of cache entries evicted due to capacity constraints.
* **Storage Tier Statistics**: Per-tier hit rate, average latency, and bandwidth utilization.

### Core Functions

#### Hash-Based Cache Lookup

Computes a hash for each chunk and queries the cache store. If a match is found, it retrieves the corresponding KV cache. Supports multi-tier lookup starting from GPU memory down to SSD.

#### Cache Loading

Transfers KV caches from storage devices into GPU memory. Uses asynchronous I/O and pinned memory to reduce transfer latency.

#### Cache Insertion

Stores newly computed KV caches in the selected storage tier. If capacity is exceeded, the eviction policy removes the least recently used caches to make space.

#### Eviction and Replacement

Implements LRU eviction with optional tier-aware demotion (e.g., evicting GPU-resident caches to CPU RAM before SSD).

#### Performance Tracking

Collects statistics on cache hit rate, average fetch latency, and I/O bandwidth usage. These metrics can be fed back to the loading controller for dynamic optimization.

---

## Integration and Coordination

### How the Fusor and Cache Store Work Together

* **Stage 1 – Cache Retrieval**: Cache store loads all available KV caches for the requested chunks into GPU memory.
* **Stage 2 – Fusion and Recompute**: Fusor merges these caches and selectively recomputes HKVD tokens layer-by-layer while pipeline-loading future layers.
* **Stage 3 – Cache Update**: Any newly generated KV caches (for previously unseen chunks) are stored back into the cache store, subject to eviction policies.
* **Stage 4 – Feedback Loop**: Performance metrics (hit rate, recompute overhead) are reported to the loading controller to refine future recompute ratios or storage device choices.
