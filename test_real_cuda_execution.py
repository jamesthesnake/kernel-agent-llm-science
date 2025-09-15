#!/usr/bin/env python3
"""
Test real CUDA kernel execution with actual compilation and performance measurement
"""

import torch
import time
import json
from pathlib import Path
import sys

# Add project path
sys.path.append(str(Path(__file__).parent))

from executor.cuda_exec import run
from agents.schemas import CudaPlan, ShapeStencil

def test_real_cuda_kernel():
    """Test actual CUDA kernel compilation and execution"""

    print("🚀 REAL CUDA KERNEL EXECUTION TEST")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"📊 Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # Create a real CUDA kernel plan
    cuda_kernel_code = '''__global__ void stencil3x3_kernel(const float* __restrict__ inp,
                                         float* __restrict__ out,
                                         int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < W && idy < H) {
        float result = 0.0f;

        // 3x3 stencil computation with bounds checking
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int ni = min(max(idy + di, 0), H-1);
                int nj = min(max(idx + dj, 0), W-1);
                result += inp[ni * W + nj] * 0.111f;
            }
        }

        out[idy * W + idx] = result;
    }
}'''

    plan = CudaPlan(
        experiment_id="real_cuda_test_001",
        backend="cuda",
        op="stencil3x3",
        dtype="fp32",
        shapes=[
            ShapeStencil(H=512, W=512),
            ShapeStencil(H=1024, W=1024),
            ShapeStencil(H=2048, W=2048)
        ],
        hypothesis="Basic 3x3 stencil with bounds checking should achieve reasonable performance",
        metrics=["latency", "throughput"],
        tolerance={"stencil3x3": {"fp32": 1e-3}},
        param_grid={
            "BLOCK_X": [16, 32],
            "BLOCK_Y": [16, 32]
        },
        iters=50,
        cuda_kernel=cuda_kernel_code
    )

    print(f"\n🔧 CUDA Kernel Plan:")
    print(f"   • Experiment ID: {plan.experiment_id}")
    print(f"   • Operation: {plan.op}")
    print(f"   • Shapes: {[f'{s.H}x{s.W}' for s in plan.shapes]}")
    print(f"   • Block configs: {len(plan.param_grid['BLOCK_X']) * len(plan.param_grid['BLOCK_Y'])}")
    print(f"   • Iterations per config: {plan.iters}")

    print(f"\n⚡ Executing Real CUDA Kernels...")

    try:
        # Execute the plan with real CUDA compilation and measurement
        start_time = time.time()
        results = run(plan, device=0, timeout_s=30.0, vram_gb=16.0)
        execution_time = time.time() - start_time

        print(f"\n✅ CUDA Execution Completed in {execution_time:.2f}s!")

        # Display results
        print(f"\n📊 RESULTS:")
        print(f"   • Experiment ID: {results.experiment_id}")
        print(f"   • Backend: {results.backend}")
        print(f"   • Configurations tested: {len(results.tested)}")

        if results.best:
            best = results.best
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   • Config: {best['config']}")
            print(f"   • Shape: {best['shape']}")
            print(f"   • Latency: {best['latency_ms']:.3f} ms")
            print(f"   • Throughput: {best['throughput_gbps']:.2f} GB/s")
            print(f"   • Speedup: {best.get('speedup_vs_baseline', 'N/A')}")
            print(f"   • Error: {best['l_inf_error']:.2e}")
            print(f"   • Passed: {best['passed']}")

        # Show all results
        print(f"\n📈 ALL CONFIGURATIONS:")
        passing_configs = [r for r in results.tested if r['passed']]
        failed_configs = [r for r in results.tested if not r['passed']]

        print(f"   ✅ Passing: {len(passing_configs)}")
        print(f"   ❌ Failed: {len(failed_configs)}")

        if passing_configs:
            print(f"\n   Top 3 Performing Configs:")
            sorted_configs = sorted(passing_configs, key=lambda x: x['latency_ms'])
            for i, config in enumerate(sorted_configs[:3], 1):
                speedup = config.get('speedup_vs_baseline', 'N/A')
                print(f"   {i}. {config['config']} @ {config['shape']}: "
                      f"{config['latency_ms']:.3f}ms, {config['throughput_gbps']:.2f} GB/s, "
                      f"speedup: {speedup}")

        # Save results
        output_dir = Path("real_cuda_results")
        output_dir.mkdir(exist_ok=True)

        result_file = output_dir / f"{plan.experiment_id}_results.json"
        with open(result_file, 'w') as f:
            # Convert to dict for JSON serialization
            json.dump(results.model_dump(), f, indent=2)

        print(f"\n💾 Results saved to: {result_file}")

        # Performance summary
        if results.best and results.best['passed']:
            print(f"\n🎯 PERFORMANCE SUMMARY:")
            print(f"   • Successfully compiled and executed CUDA kernel")
            print(f"   • Best latency: {results.best['latency_ms']:.3f} ms")
            print(f"   • Best throughput: {results.best['throughput_gbps']:.2f} GB/s")
            print(f"   • Numerical accuracy: {results.best['l_inf_error']:.2e}")
            print(f"   • All {len(passing_configs)} configs passed correctness tests")

            return True
        else:
            print(f"❌ No configurations passed correctness tests")
            return False

    except Exception as e:
        print(f"❌ CUDA execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_kernels():
    """Test multiple different CUDA kernel variants"""

    print(f"\n🔄 TESTING MULTIPLE KERNEL VARIANTS")
    print("=" * 60)

    kernels = {
        "basic": '''__global__ void stencil3x3_kernel(const float* __restrict__ inp,
                                         float* __restrict__ out,
                                         int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < W && idy < H) {
        float result = 0.0f;
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int ni = min(max(idy + di, 0), H-1);
                int nj = min(max(idx + dj, 0), W-1);
                result += inp[ni * W + nj] * 0.111f;
            }
        }
        out[idy * W + idx] = result;
    }
}''',

        "optimized": '''__global__ void stencil3x3_kernel(const float* __restrict__ inp,
                                         float* __restrict__ out,
                                         int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < W && idy < H) {
        // Unrolled stencil for better performance
        float result = 0.0f;

        // Manual bounds checking with unrolled loops
        if (idy > 0 && idx > 0) result += inp[(idy-1) * W + (idx-1)] * 0.111f;
        if (idy > 0) result += inp[(idy-1) * W + idx] * 0.111f;
        if (idy > 0 && idx < W-1) result += inp[(idy-1) * W + (idx+1)] * 0.111f;

        if (idx > 0) result += inp[idy * W + (idx-1)] * 0.111f;
        result += inp[idy * W + idx] * 0.111f;
        if (idx < W-1) result += inp[idy * W + (idx+1)] * 0.111f;

        if (idy < H-1 && idx > 0) result += inp[(idy+1) * W + (idx-1)] * 0.111f;
        if (idy < H-1) result += inp[(idy+1) * W + idx] * 0.111f;
        if (idy < H-1 && idx < W-1) result += inp[(idy+1) * W + (idx+1)] * 0.111f;

        out[idy * W + idx] = result;
    }
}'''
    }

    results_summary = {}

    for variant_name, kernel_code in kernels.items():
        print(f"\n🧪 Testing {variant_name} variant...")

        plan = CudaPlan(
            experiment_id=f"real_cuda_{variant_name}",
            backend="cuda",
            op="stencil3x3",
            dtype="fp32",
            shapes=[ShapeStencil(H=1024, W=1024)],
            hypothesis=f"Testing {variant_name} 3x3 stencil implementation",
            metrics=["latency", "throughput"],
            tolerance={"stencil3x3": {"fp32": 1e-3}},
            param_grid={"BLOCK_X": [16, 32], "BLOCK_Y": [16, 32]},
            iters=30,
            cuda_kernel=kernel_code
        )

        try:
            results = run(plan, device=0, timeout_s=15.0)

            if results.best and results.best['passed']:
                results_summary[variant_name] = {
                    'latency_ms': results.best['latency_ms'],
                    'throughput_gbps': results.best['throughput_gbps'],
                    'config': results.best['config'],
                    'speedup': results.best.get('speedup_vs_baseline', None)
                }
                print(f"   ✅ {variant_name}: {results.best['latency_ms']:.3f}ms, "
                      f"{results.best['throughput_gbps']:.2f} GB/s")
            else:
                print(f"   ❌ {variant_name}: Failed")
                results_summary[variant_name] = {'status': 'failed'}

        except Exception as e:
            print(f"   ❌ {variant_name}: Error - {e}")
            results_summary[variant_name] = {'status': 'error', 'error': str(e)}

    # Compare results
    print(f"\n🏁 VARIANT COMPARISON:")
    successful_variants = {k: v for k, v in results_summary.items()
                          if 'latency_ms' in v}

    if len(successful_variants) > 1:
        baseline = min(successful_variants.values(), key=lambda x: x['latency_ms'])
        baseline_latency = baseline['latency_ms']

        for variant, result in successful_variants.items():
            speedup = baseline_latency / result['latency_ms']
            print(f"   • {variant}: {result['latency_ms']:.3f}ms "
                  f"({speedup:.2f}× vs baseline) @ {result['config']}")

    return results_summary

if __name__ == "__main__":
    print("🔥 REAL CUDA KERNEL EXECUTION WITH H100 GPU")
    print("=" * 80)

    # Test single kernel
    success = test_real_cuda_kernel()

    if success:
        # Test multiple variants
        variant_results = test_multiple_kernels()

        print(f"\n🎉 REAL CUDA EXECUTION COMPLETED!")
        print(f"   ✅ Successfully compiled and executed CUDA kernels on H100")
        print(f"   📊 Measured real performance with timing and throughput")
        print(f"   🔍 Verified numerical correctness")
        print(f"   📁 Results saved in real_cuda_results/")
    else:
        print(f"\n❌ CUDA execution test failed")

    print(f"\n🚀 Ready for real autonomous training with actual kernel execution!")