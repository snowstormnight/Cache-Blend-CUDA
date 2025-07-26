# Loading Controller Module - Two-Part Design

## Part 1: Recompute Ratio Optimization (Fixed Storage Device)

### Description
Given a specific storage device that has been selected for use, this component determines the optimal percentage of tokens to recompute such that the recomputation can be completely hidden by the KV cache loading time, thereby incurring no additional delay to the time-to-first-token (TTFT).

### Input Description
- **Model Configuration**: The specific LLM model being used (Mistral-7B, Yi-34B, or Llama-70B) with its complete architectural specifications including number of layers, hidden dimensions, attention heads, and quantization level (fp16, int8, or int4).
- **Context Length**: The total number of tokens in the input context that needs to be processed.
- **Selected Storage Device**: A specific storage device that has already been chosen, characterized by its type (GPU HBM, CPU RAM, NVMe SSD, etc.), read bandwidth in GB/s, and access latency in milliseconds.
- **Minimum Quality Ratio**: The empirically-determined minimum recompute ratio (typically 15%) below which generation quality degrades unacceptably.
- **Prefill Profile Data**: Pre-collected timing data mapping model-context length pairs to measured prefill times, used for accurate delay estimation.

### Output Description
- **Optimal Recompute Ratio**: The calculated percentage of tokens that should be recomputed per layer to maximize efficiency while maintaining quality.
- **Achieves Perfect Hiding**: Boolean indicating whether the recomputation can be completely hidden by loading time (true when recompute delay ≤ loading delay).
- **Expected Total Delay**: The maximum of loading and recompute delays, representing the actual TTFT contribution.
- **Quality Score**: Predicted generation quality (F1 or Rouge-L) at the selected recompute ratio.

### Core Functions

#### Calculate Loading Delay for Given Device
This function computes the time required to load the KV cache from the specified storage device to GPU memory. It calculates the total KV cache size based on the model architecture and context length, then divides by the device's bandwidth to get transfer time. Access latency and per-layer coordination overhead are added to get the total loading delay.

#### Calculate Recompute Delay for Given Ratio
This function estimates the time to selectively recompute a specified percentage of tokens. It uses the profiled full prefill time for the model and context length, then scales it by the recompute ratio. An overhead factor (typically 5%) is applied to account for the selective recomputation mechanism's complexity.

#### Find Crossover Ratio
This function uses binary search to find the exact recompute ratio where the recompute delay equals the loading delay. This represents the maximum amount of recomputation that can be performed without adding extra delay. The search maintains precision to 0.1% and ensures the result never falls below the minimum quality threshold.

#### Validate and Adjust Ratio
This function ensures the computed ratio meets all constraints. If the crossover ratio is below the minimum quality threshold, it returns the minimum threshold instead. It also validates that the ratio is achievable given the model's architecture and the available computational resources.

## Part 2: Storage Device Selection (Fixed Recompute Ratio)

### Description
Given a fixed recompute ratio (typically 15% for maintaining generation quality), this component selects the most cost-effective storage device from available options that ensures no additional delay is added to the inference process.

### Input Description
- **Model Configuration**: Same as Part 1 - complete specifications of the LLM model being used.
- **Context Length**: The total number of tokens to be processed.
- **Fixed Recompute Ratio**: A predetermined recompute percentage, typically 15% based on empirical quality studies.
- **Available Storage Devices**: A list of all storage options, each with specifications including type, bandwidth, latency, capacity, available space, and cost per GB-hour (for cloud storage).
- **Optimization Preference**: Whether to optimize for minimum cost or minimum latency.
- **Storage Duration**: Expected time the KV cache will be stored, used for cost calculations.
- **Prefill Profile Data**: Same profiling data as Part 1.

### Output Description
- **Selected Storage Device**: The chosen storage device that best meets the optimization criteria.
- **Expected Performance Metrics**: Loading delay, recompute delay, and total delay for the selected device.
- **Cost Analysis**: Hourly storage cost for the selected device (zero for local storage).
- **Zero Extra Delay Achievement**: Boolean confirming that the recomputation is hidden by loading time.
- **Alternative Options**: Ranked list of other viable storage devices with their respective performance characteristics.

### Core Functions

#### Evaluate Device Feasibility
This function checks each available storage device for basic compatibility. It verifies sufficient available capacity for the KV cache size, ensures the device is accessible from the compute node, and confirms the device's bandwidth is sufficient for reasonable loading times.

#### Calculate Performance for Each Device
For each feasible device, this function computes the expected loading delay based on the device's bandwidth and latency. It then calculates the recompute delay for the fixed ratio and determines whether perfect hiding is achieved (recompute delay ≤ loading delay).

#### Calculate Storage Cost
This function computes the economic cost of storing the KV cache on each device. For cloud storage, it considers the size, duration, and per-GB-hour pricing. For local storage, it may assign zero cost or calculate opportunity cost. The function can account for additional factors like data transfer fees or redundancy overhead.

#### Rank and Select Devices
This function implements the selection logic based on the optimization preference. When optimizing for cost, it filters devices that achieve zero extra delay, then selects the cheapest option. When optimizing for performance, it simply chooses the device with minimum total delay. The ranking considers both primary criteria and secondary factors like reliability and availability.

#### Generate Alternative Configurations
This function prepares a list of alternative storage options ranked by their suitability. For each alternative, it may calculate the optimal recompute ratio specific to that device, providing flexibility if the primary choice becomes unavailable or if requirements change.

## Integration and Coordination

### How the Two Parts Work Together

#### Sequential Operation Mode
In some deployments, Part 1 operates first with a default or previously selected storage device to determine the optimal recompute ratio. If the result is unsatisfactory (e.g., ratio too high for quality), Part 2 is invoked to find a better storage device that allows a lower recompute ratio.

#### Parallel Evaluation Mode
For comprehensive optimization, both parts can operate in parallel. Part 2 evaluates all devices with the minimum quality ratio, while Part 1 calculates optimal ratios for each device. The results are combined to find the global optimum considering both performance and cost.

#### Dynamic Adaptation Mode
During runtime, the controller can switch between parts based on changing conditions. If storage device performance degrades, Part 1 recalculates the recompute ratio. If quality requirements change, Part 2 finds a new device that supports the updated ratio.

### Shared Components

#### Delay Calculation Engine
Both parts use the same underlying functions for calculating loading and recompute delays. This ensures consistency and allows for code reuse. The engine maintains device performance profiles and model characteristics.

#### Quality Estimation Model
The relationship between recompute ratio and generation quality is used by both parts - Part 1 to enforce minimum thresholds, and Part 2 to predict quality for the fixed ratio.

#### Profiling Data Management
Both parts access the same profiling database containing prefill times and performance characteristics. This data is collected offline and updated periodically to reflect hardware or model changes.