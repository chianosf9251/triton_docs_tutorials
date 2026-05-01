"""
- basics
- test
- benchmark
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
print(f"Using device: {DEVICE}")


# this `triton.jit` decorator tells Triton to compile this function into GPU code
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """the BLOCK_SIZE is a compile-time variable (rather than run-time),
    meaning that every time a different value for BLOCK_SIZE is passed in you're actually
    creating an entirely separate kernel. I may sometimes refer to arguments with this
    designation as "meta-parameters"
    """

    PID = tl.program_id(axis=0)
    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Problem: if the last block is not a multiple of BLOCK_SIZE, then we will have out-of-bounds accesses
    # Solution: create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements

    # load data from DRAM/VRAM/global GPU memory/high-bandwidth memory onto SRAM/on-chip memory
    x = tl.load(x_ptr + offsets, mask=mask, other=None)  # shape (BLOCK_SIZE)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)  # shape (BLOCK_SIZE)

    # perform the operation on SRAM
    output = x + y

    # write data back to DRAM/VRAM/global GPU memory/high-bandwidth memory
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # pre-allocate the output
    output = torch.empty_like(x)

    # check tensors are on the same device
    assert (
        x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    ), f"DEVICE: {DEVICE}, x.device: {x.device}, y.device: {y.device}, output.device: {output.device}"

    # get the number of elements in the vectors
    n_elements = output.numel()

    # define the launch grid (how many programs/blocks we launch)
    # we use a lambda function to define the grid because it allows us to pass in meta-parameters
    # meta-parameters are parameters that are passed in at compile-time, not runtime
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # (4, )
    # BLOCK_SIZE is a meta-parameter
    # cdiv(m, n) = (m + (n - 1)) // n

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # BLOCK_SIZE is a meta-parameter that is passed in at compile-time, not runtime
    # we use a heuristic choice of 1024 for BLOCK_SIZE
    # it's a power of 2 (efficient for memory access patterns)
    # it's large enough to hide memory latency
    # it's small enough to allow multiple blocks to run concurrently on a GPU
    # in a later lesson we'll learn better methods than heuristics

    return output


######### Test Function #########
def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    """
    Here is where we test the wrapper function and kernel that we wrote
    above to ensure all our values are correct, using pytorch as the
    correct answer to compare against, where the
    atol: absolute tolerance, rtol: relative tolerance
    """
    # Create test data, x, y ∈ ℝ^size
    torch.manual_seed(0)  # seed for reproducibility
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)
    # Run triton kernel & pytorch reference implementation
    z_tri = add(x, y)
    z_ref = x + y
    # Compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 24, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    # create input data
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95] # quantiles tells matplotlib what confidence intervals to plot
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path=".", print_data=False)