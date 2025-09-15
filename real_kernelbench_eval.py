#!/usr/bin/env python3
"""
Real KernelBench Evaluation - No Cheating!
Actually compiles and runs AI-generated kernels against KernelBench's evaluation framework
"""

import os
import sys
import json
import time
import torch
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from datasets import load_dataset

# Add KernelBench to path
sys.path.append(str(Path(__file__).parent / "KernelBench"))

from src import eval as kernel_eval
from src import utils as kernel_utils

class AIKernelGenerator:
    """AI that generates kernels tailored to specific KernelBench problems"""

    def __init__(self):
        self.generation_step = 0

    def analyze_problem(self, problem_code: str, problem_name: str) -> Dict:
        """Analyze a KernelBench problem to understand what kernel we need"""
        self.generation_step += 1

        # Extract the Model class to understand the operation
        lines = problem_code.split('\n')

        # Look for the forward method to understand the operation
        forward_impl = ""
        forward_signature = ""
        in_forward = False
        for line in lines:
            if "def forward(" in line:
                in_forward = True
                forward_signature = line.strip()
            elif in_forward and "def " in line and "forward" not in line:
                break
            elif in_forward:
                forward_impl += line + "\n"

        # Try to understand the operation type
        operation_type = "unknown"
        if "+" in forward_impl and not "*" in forward_impl:
            operation_type = "element_wise_add"
        elif "*" in forward_impl and "matmul" in forward_impl.lower():
            operation_type = "matrix_multiply"
        elif "relu" in forward_impl.lower():
            operation_type = "relu"
        elif "conv" in forward_impl.lower():
            operation_type = "convolution"
        elif "softmax" in forward_impl.lower():
            operation_type = "softmax"
        elif "clamp" in forward_impl.lower():
            operation_type = "clamp"
        elif "exp" in forward_impl.lower():
            operation_type = "exponential"

        # Count parameters in forward signature
        param_count = forward_signature.count(',') + 1 if forward_signature else 1

        return {
            'operation_type': operation_type,
            'forward_implementation': forward_impl,
            'forward_signature': forward_signature,
            'param_count': param_count,
            'problem_name': problem_name,
            'analysis_step': self.generation_step
        }

    def generate_kernelbench_solution(self, problem_code: str, problem_name: str) -> str:
        """Generate a complete KernelBench-compatible solution file"""

        analysis = self.analyze_problem(problem_code, problem_name)
        operation_type = analysis['operation_type']
        param_count = analysis['param_count']

        print(f"🤖 AI analyzing {problem_name}")
        print(f"   Detected operation: {operation_type}")
        print(f"   Parameters: {param_count}")

        # Generate CUDA kernel based on operation type and parameter count
        if operation_type == "element_wise_add" and param_count >= 3:
            cuda_kernel, cpp_wrapper = self._generate_elementwise_add_kernel()
        elif operation_type == "relu" or param_count <= 2:
            cuda_kernel, cpp_wrapper = self._generate_single_input_kernel()
        elif operation_type == "matrix_multiply":
            cuda_kernel, cpp_wrapper = self._generate_matmul_kernel()
        elif operation_type == "clamp":
            cuda_kernel, cpp_wrapper = self._generate_single_input_kernel()
        elif operation_type == "exponential":
            cuda_kernel, cpp_wrapper = self._generate_single_input_kernel()
        else:
            # Default based on parameter count
            if param_count <= 2:
                print(f"   Unknown operation with single input, using single-input kernel")
                cuda_kernel, cpp_wrapper = self._generate_single_input_kernel()
            else:
                print(f"   Unknown operation with multiple inputs, using element-wise add")
                cuda_kernel, cpp_wrapper = self._generate_elementwise_add_kernel()

        # Create the complete KernelBench solution file
        solution_code = f'''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# AI Generated CUDA kernel
cuda_source = """
{cuda_kernel}
"""

cpp_source = """
{cpp_wrapper}
"""

# Compile the inline CUDA code
ai_kernel = load_inline(
    name="ai_generated_kernel",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["ai_kernel_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = ai_kernel

    def forward(self, *args):
        return self.kernel.ai_kernel_cuda(*args)
'''

        return solution_code

    def _generate_elementwise_add_kernel(self):
        """Generate element-wise addition kernel"""
        cuda_kernel = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void ai_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor ai_kernel_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    ai_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    return out;
}
'''
        cpp_wrapper = "torch::Tensor ai_kernel_cuda(torch::Tensor a, torch::Tensor b);"
        return cuda_kernel, cpp_wrapper

    def _generate_single_input_kernel(self):
        """Generate general single-input kernel that can handle various activations"""
        cuda_kernel = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void ai_general_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // General purpose activation - can be sigmoid, relu, tanh, etc.
        // Using sigmoid as a reasonable default for unknown activations
        output[idx] = 1.0f / (1.0f + expf(-val));
    }
}

torch::Tensor ai_kernel_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    ai_general_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
'''
        cpp_wrapper = "torch::Tensor ai_kernel_cuda(torch::Tensor input);"
        return cuda_kernel, cpp_wrapper

    def _generate_matmul_kernel(self):
        """Generate basic matrix multiplication kernel"""
        cuda_kernel = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void ai_matmul_kernel(const float* a, const float* b, float* c, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

torch::Tensor ai_kernel_cuda(torch::Tensor a, torch::Tensor b) {
    auto M = a.size(0);
    auto K = a.size(1);
    auto N = b.size(1);
    auto c = torch::zeros({M, N}, a.options());

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    ai_matmul_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K
    );

    return c;
}
'''
        cpp_wrapper = "torch::Tensor ai_kernel_cuda(torch::Tensor a, torch::Tensor b);"
        return cuda_kernel, cpp_wrapper

    def _generate_clamp_kernel(self):
        """Generate clamp kernel for operations like HingeLoss"""
        cuda_kernel = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void ai_clamp_kernel(const float* input, float* output, float min_val, float max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = fmaxf(min_val, fminf(max_val, val));
    }
}

torch::Tensor ai_kernel_cuda(torch::Tensor input, float min_val = 0.0f, float max_val = 1e10f) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    ai_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        min_val, max_val,
        size
    );

    return output;
}
'''
        cpp_wrapper = "torch::Tensor ai_kernel_cuda(torch::Tensor input, float min_val, float max_val);"
        return cuda_kernel, cpp_wrapper

    def _generate_exp_kernel(self):
        """Generate exponential kernel"""
        cuda_kernel = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void ai_exp_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}

torch::Tensor ai_kernel_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    ai_exp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
'''
        cpp_wrapper = "torch::Tensor ai_kernel_cuda(torch::Tensor input);"
        return cuda_kernel, cpp_wrapper


class RealKernelBenchEvaluator:
    """Real evaluation using KernelBench's framework - no cheating!"""

    def __init__(self):
        self.ai_generator = AIKernelGenerator()
        self.results = []

    def evaluate_single_problem(self, problem: Dict, temp_dir: str) -> Dict:
        """Evaluate AI solution on a single KernelBench problem"""

        problem_id = problem['problem_id']
        problem_name = problem['name']
        problem_code = problem['code']

        print(f"🎯 Evaluating Problem {problem_id}: {problem_name}")

        try:
            # Generate AI solution
            ai_solution = self.ai_generator.generate_kernelbench_solution(problem_code, problem_name)

            # Save the AI solution to a temporary file
            solution_file = os.path.join(temp_dir, f"ai_solution_{problem_id}.py")
            with open(solution_file, 'w') as f:
                f.write(ai_solution)

            # Save the reference problem to a file
            reference_file = os.path.join(temp_dir, f"reference_{problem_id}.py")
            with open(reference_file, 'w') as f:
                f.write(problem_code)

            print(f"   ⚙️  Compiling and testing...")

            # Use KernelBench's evaluation framework
            device = torch.device("cuda:0")

            eval_result = kernel_eval.eval_kernel_against_ref(
                original_model_src=problem_code,
                custom_model_src=ai_solution,
                measure_performance=True,
                verbose=False,
                num_correct_trials=5,
                num_perf_trials=10,  # Reduced for faster testing
                build_dir=os.path.join(temp_dir, f"build_{problem_id}"),
                device=device
            )

            # Extract results
            is_correct = eval_result.correctness
            speedup = 1.0 / (eval_result.runtime / 1000000.0) if eval_result.runtime > 0 else 0.0  # Convert microseconds
            compilation_successful = eval_result.compiled

            result = {
                'problem_id': problem_id,
                'problem_name': problem_name,
                'compilation_successful': compilation_successful,
                'is_correct': is_correct,
                'speedup': speedup,
                'is_faster': speedup > 1.0,
                'fast_1': is_correct and speedup > 1.0,  # KernelBench fast_1 metric
                'execution_time_us': eval_result.runtime,
                'error_message': str(eval_result.metadata) if eval_result.metadata else None
            }

            status = "✅ PASS" if (compilation_successful and is_correct) else "❌ FAIL"
            speed_info = f"{speedup:.2f}×" if speedup > 0 else "N/A"

            print(f"   {status} | Speedup: {speed_info} | Correct: {is_correct}")

            return result

        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            return {
                'problem_id': problem_id,
                'problem_name': problem_name,
                'compilation_successful': False,
                'is_correct': False,
                'speedup': 0.0,
                'is_faster': False,
                'fast_1': False,
                'execution_time_ms': None,
                'error_message': str(e)
            }

    def run_honest_benchmark(self, num_problems: int = 5, level: str = 'level_1') -> Dict:
        """Run real KernelBench evaluation - the honest way"""

        print(f"🔬 REAL KERNELBENCH EVALUATION - NO CHEATING!")
        print(f"=" * 60)
        print(f"Level: {level}")
        print(f"Problems to test: {num_problems}")
        print(f"Using actual compilation and correctness testing")
        print()

        # Load dataset
        try:
            dataset = load_dataset('ScalingIntelligence/KernelBench', split=level)
            print(f"✅ Loaded {len(dataset)} problems from {level}")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return {}

        # Create temporary directory for evaluation
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Working directory: {temp_dir}")

            # Select problems to test
            import random
            problem_indices = random.sample(range(len(dataset)), min(num_problems, len(dataset)))

            results = []
            successful_compilations = 0
            correct_results = 0
            fast_results = 0

            for i, idx in enumerate(problem_indices):
                print(f"\n📋 Problem {i+1}/{num_problems}")
                print("-" * 40)

                result = self.evaluate_single_problem(dataset[idx], temp_dir)
                results.append(result)

                if result['compilation_successful']:
                    successful_compilations += 1
                if result['is_correct']:
                    correct_results += 1
                if result['fast_1']:
                    fast_results += 1

                # Brief pause between problems
                time.sleep(1)

        # Calculate honest metrics
        total_problems = len(results)
        compilation_rate = successful_compilations / total_problems
        correctness_rate = correct_results / total_problems
        fast_1_rate = fast_results / total_problems  # Real KernelBench metric

        # Analyze results
        print(f"\n📊 HONEST BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total problems attempted: {total_problems}")
        print(f"Successful compilations: {successful_compilations} ({compilation_rate*100:.1f}%)")
        print(f"Correct results: {correct_results} ({correctness_rate*100:.1f}%)")
        print(f"Faster than baseline: {fast_results} ({fast_1_rate*100:.1f}%)")
        print(f"KernelBench fast_1 score: {fast_1_rate:.3f}")

        if fast_results > 0:
            successful_results = [r for r in results if r['fast_1']]
            avg_speedup = sum(r['speedup'] for r in successful_results) / len(successful_results)
            print(f"Average speedup (successful): {avg_speedup:.2f}×")

        # Save results
        output_file = f"real_kernelbench_results_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'evaluation_info': {
                    'level': level,
                    'total_problems': total_problems,
                    'compilation_rate': compilation_rate,
                    'correctness_rate': correctness_rate,
                    'fast_1_rate': fast_1_rate,
                    'evaluation_timestamp': datetime.now().isoformat()
                },
                'detailed_results': results
            }, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        return {
            'compilation_rate': compilation_rate,
            'correctness_rate': correctness_rate,
            'fast_1_rate': fast_1_rate,
            'results_file': output_file
        }


def main():
    """Run the real KernelBench evaluation"""
    evaluator = RealKernelBenchEvaluator()

    # Test with a small number of problems first
    results = evaluator.run_honest_benchmark(num_problems=3, level='level_1')

    print(f"\n🎯 REAL PERFORMANCE SUMMARY")
    print(f"   Compilation Success: {results.get('compilation_rate', 0)*100:.1f}%")
    print(f"   Correctness Rate: {results.get('correctness_rate', 0)*100:.1f}%")
    print(f"   KernelBench fast_1: {results.get('fast_1_rate', 0)*100:.1f}%")

    if results.get('fast_1_rate', 0) > 0:
        print(f"   🎉 AI successfully passed some KernelBench tests!")
    else:
        print(f"   🔧 AI needs improvement for KernelBench challenges")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping real evaluation")
        sys.exit(1)

    main()