"""
Plant Disease Classification - ONNX Model Benchmark Script

This script benchmarks the performance of ONNX models.
It measures latency, FPS, RAM and CPU usage for each model with different thread counts.
Results are saved to a CSV file for easy analysis.
"""

import time
import psutil
import numpy as np
import onnxruntime as ort
import pandas as pd
import os
import gc

# ==================== Configuration ====================

# Directory containing ONNX models
checkpoint_dir = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/onnx_model/plant_village"

# Dictionary mapping model names to their ONNX file paths
MODEL_PATHS = {
    "VGG16": f"{checkpoint_dir}/vgg16.onnx",
    "MobileNetV3-Small": f"{checkpoint_dir}/mobilenetv3_small.onnx",
    "MobilePlantViT": f"{checkpoint_dir}/mobileplantvit.onnx",
    "DenseNet121": f"{checkpoint_dir}/densnet121.onnx",
    "ResNet50": f"{checkpoint_dir}/resnet50.onnx",
    "SqueezeNetV2": f"{checkpoint_dir}/squezzenetv2.onnx",
    "ShuffleNetV2": f"{checkpoint_dir}/shufflenetv2.onnx",
}

# Input shape: (batch_size=1, channels=3, height=224, width=224)
INPUT_SHAPE = (1, 3, 224, 224)

# Number of warmup runs to initialize cache (results are discarded)
N_WARMUP = 15

# Number of actual runs for benchmarking data collection
N_RUNS = 100

# List of CPU thread counts to test (1=single-threaded, 4=multi-threaded)
THREAD_SETTINGS = [1, 4]

# Output CSV filename for benchmark results
OUTPUT_CSV = "pi4b_onnx_benchmark.csv"



# ==================== ONNX Session Creation ====================

def create_session(model_path, threads):
    """
    Create an ONNX InferenceSession with specified thread configuration.
    
    Args:
        model_path (str): Path to the ONNX model file
        threads (int): Number of CPU threads for inference
    
    Returns:
        ort.InferenceSession: Configured inference session
    """
    # Create SessionOptions for detailed configuration
    session_options = ort.SessionOptions()
    
    # Number of threads for intra-op parallelism (e.g., matrix multiplication)
    session_options.intra_op_num_threads = threads
    
    # Number of threads for inter-op parallelism (independent operations)
    session_options.inter_op_num_threads = 1
    
    # Execution mode: ORT_SEQUENTIAL = sequential execution (no parallelism)
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Graph optimization: ORT_ENABLE_ALL = enable all available optimizations
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Create session using CPU execution provider only
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )

# ==================== Benchmarking Function ====================

def benchmark(model_name, model_path, threads):
    """
    Benchmark an ONNX model by running it multiple times and measuring performance metrics.
    
    Args:
        model_name (str): Model name (for display and results)
        model_path (str): Path to the .onnx file
        threads (int): Number of CPU threads to use
    
    Returns:
        dict: Dictionary containing metrics:
              Model, Threads, Latency_ms/img, FPS, Peak_RAM_MB, CPU_Usage_%
    """
    print(f"\n▶ {model_name} | Threads={threads}")

    # Create session with specified thread configuration
    session = create_session(model_path, threads)
    
    # Get input layer name (first layer of the model)
    input_name = session.get_inputs()[0].name
    
    # Create dummy input: random array with shape (1, 3, 224, 224), dtype float32
    dummy_input = np.random.rand(*INPUT_SHAPE).astype(np.float32)

    # ===== WARMUP PHASE =====
    # Run the model N_WARMUP times to:
    # - Initialize CPU cache
    # - Allow ONNX runtime to optimize kernels
    for _ in range(N_WARMUP):
        session.run(None, {input_name: dummy_input})

    # Get current process for RAM monitoring
    process = psutil.Process(os.getpid())

    # List to store latency (inference time) for each run
    latencies = []
    
    # List to store CPU usage for each run
    cpu_list = []
    
    # Variable to track peak RAM usage
    peak_ram = 0

    # ===== BENCHMARK PHASE =====
    for _ in range(N_RUNS):
        gc.collect()

        # Reset system-wide CPU usage counter
        psutil.cpu_percent(interval=None)

        # Record start time
        start = time.perf_counter()

        # Run inference
        session.run(None, {input_name: dummy_input})

        # Record end time
        end = time.perf_counter()

        # Get system-wide CPU usage during inference
        cpu = psutil.cpu_percent(interval=None)

        # Store latency in milliseconds
        latencies.append((end - start) * 1000)

        # Store CPU usage
        cpu_list.append(cpu)

        # Measure RAM usage
        ram_mb = process.memory_info().rss / 1024 / 1024
        peak_ram = max(peak_ram, ram_mb)

    # ===== CALCULATE METRICS =====
    # Average latency in milliseconds
    avg_latency = np.mean(latencies)
    
    # FPS = 1000ms / latency_ms (images processed per second)
    fps = 1000 / avg_latency

    # Return results dictionary
    return {
        "Model": model_name,
        "Threads": threads,
        "Latency_ms/img": round(avg_latency, 2),
        "FPS": round(fps, 2),
        "Peak_RAM_MB": round(peak_ram, 1),
        "CPU_Usage_%": round(np.mean(cpu_list), 1),
    }


# ==================== MAIN EXECUTION ====================

# List to store benchmark results for all models and thread settings
results = []

# Loop through each model
for model_name, model_path in MODEL_PATHS.items():
    # Loop through each thread setting
    for num_threads in THREAD_SETTINGS:
        # Run benchmark and store results
        results.append(benchmark(model_name, model_path, num_threads))

# Convert results list to Pandas DataFrame
df = pd.DataFrame(results)

# Save DataFrame to CSV file
df.to_csv(OUTPUT_CSV, index=False)

# Display completion message and results
print("\n✅ DONE")
print(df)
print(f"\n📁 Saved to {OUTPUT_CSV}")
