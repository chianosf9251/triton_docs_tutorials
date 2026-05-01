"""
- reduce memory reads/writes by fusing a kernel
- how to get GPU sepcifications
- some more details on GPU architecture
- how to define meta-parameters using GPU-specific attributes and rough heuristics
- more about masking and how to choose the value of extra entries when masking
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
print(f"Using device: {DEVICE}")


# step 1
def naive_softmax(x):
    # assume input size is (M,N)

    # reads MN elements and writes M elements
    x_max = x.max(dim=1)[0]  # shape (M)
    # read MN + M elements, subtraction is MN flops, and write MN elements back from SRAM to DRAM
    z = x - x_max[:, None]  # shape (M,N) - shape (M, 1) = shape (M, N)
    # reading MN elements and writing MN elements
    numerator = torch.exp(z)  # shape (M,N)
    # read MN elements, then MN flops, wriet back M elements
    denominator = numerator.sum(dim=1)  # shape (M, N) -> shape (M)
    # read MN + M elements, division is MN flops, and write MN elements back from SRAM to DRAM
    out = numerator / denominator[:, None]  # shape (M, N) / shape (M, 1) = shape (M, N)

    # in total we did 8MN + 4M memory operations (very slow!!!)
    return out


# step 4
@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # shape (M, N)
    # BLOCK_SIZE is the smallest power of 2 greater than N (n_cols)

    PID = tl.program_id(0)

    row_step = tl.num_programs(0)
    # if 4 programs, then row_step = 4, and each program will process every 4th row
    # if n_rows = 6
    # pid would get row 0
    # pid 1 row 1
    # pid 2 row 2
    # pid 3 row 3
    # once they're done with those rows, they would wrap around and process the remaining rows
    # pid 0 += row_step -> row 4
    # pid 1 += row_step -> row 5

    for row_idx in tl.range(PID, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(
            0, BLOCK_SIZE
        )  # BLOCK_SIZE is the smallest power of 2 greater than n_cols
        input_ptrs = row_start_ptr + col_offsets  # shape (BLOCK_SIZE,)
        mask = (
            col_offsets < n_cols
        )  # shape (BLOCK_SIZE,), True for valid columns, False for out-of-bounds columns
        # here is the only time we call tl.load in the entire kernel, and we load BLOCK_SIZE elements at a time, which is the maximum number of columns we can process in SRAM at once
        row = tl.load(
            input_ptrs, mask=mask, other=-float("inf")
        )  # shape (BLOCK_SIZE,), out-of-bounds columns are set to -inf
        # n_cols = 3
        # BLOCL_SIZE = 4 (we have 1 extra column that is out-of-bounds)
        # col_offsets = [0, 1, 2, 3]
        # input_ptrs = row_start_ptr + col_offsets = [row_start_ptr, row_start_ptr + 1, row_start_ptr + 2, row_start_ptr + 3]
        # mask = col_offsets < n_cols = [True, True, True, False]
        # row = tl.load(input_ptrs, mask=mask, other=-float("inf")) = [row[0], row[1], row[2], -inf]
        row_mimus_max = row - tl.max(
            row, axis=0
        )  # shape (BLOCK_SIZE,) - shape (1,) = shape (BLOCK_SIZE,)
        numerator = tl.exp(
            row_mimus_max
        )  # shape (BLOCK_SIZE,), the out-of-bounds columns will be exp(-inf) = 0, so they won't contribute to the sum in the denominator
        denominator = tl.sum(numerator, axis=0)  # shape (1)
        softmax_output = (
            numerator / denominator
        )  # shape (BLOCK_SIZE,) / shape (1,) = shape (BLOCK_SIZE,)

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


# step 3
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
print(f"Number of Streaming Multiprocessors (SMs): {NUM_SM}")
print(f"Number of Registers: {NUM_REGS}")
print(f"Total SRAM per SM: {TOTAL_SRAM_PER_SM}")
print(f"Warp Size: {WARP_SIZE}")

# Using device: cuda:0
# Number of Streaming Multiprocessors (SMs): 80
# Number of Registers: 65536
# Total SRAM per SM: 101376
# Warp Size: 32

def softmax(x):
    assert x.ndim == 2
    assert x.is_contiguous()
    n_rows, n_cols = x.shape  # shape (M, N)
    # assume every row of x fits in SRAM
    # the block size is the smallest power of 2 greater than the number of columns in x
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # a trick we can use is to ask the compiler to use more threads per row by
    #  increasing the number of warps (`num_warps`) over which each row is distributed.
    # for now these settings are just a heuristic
    # you will see in the next tutorial how to auto-tune this value in a more natural way
    #   so you don't have to come up with manual heuristics yourself
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2  # heuristic

    y = torch.empty_like(x)

    # warm-up
    kernel = _softmax_kernel.warmup(
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )
    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared

    reg_occupancy = NUM_REGS // (n_regs_per_program * WARP_SIZE * num_warps)
    # NUM_REGS = 65536
    # each program uses n_regs_per_program=32 * WARP_SIZE=32 * num_warps=8 registers total
    # therefore we can fit reg_occupancy programs per SM
    # ex. 65536 // (32 * 32 * 8) = 8 programs per SM (assuming num_warps=8)
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    # each program needs sram_needed_per_program bytes of SRAM
    # therefore we can fit sram_occupancy programs per SM
    # ex. 16384 // 1024 = 16 programs per SM (assuming sram_needed_per_program=1024)
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM * programs_per_sm, n_rows)  # 1 row per program

    grid = (num_programs, 1, 1)

    kernel[grid](x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)

    # x.stride() for each dimension tells us how many entries in memory a pointer needs to move forward in order
    # x is shape (M, N), so x.stride(0) = N and x.stride(1) = 1
    # z is shape(B, N, D)
    # z.stride() would be (N*D, D, 1)
    return y


# step 2
def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    assert type(size) is tuple and len(size) == 2
    torch.manual_seed(0)
    x = torch.randn(size[0], size[1], device=device)
    # tri stands for triton implementation
    z_tri = softmax(x)
    # compare with pytorch reference implementation
    z_ref = torch.softmax(x, axis=1)
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")

# step 5
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals = [128*i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
        
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    # test_softmax_kernel((1024, 512))
    # test_softmax_kernel((1024, 1023))
    # test_softmax_kernel((1024, 1024))
    # test_softmax_kernel((1024, 1025))
    test_softmax_kernel((1823, 781))
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path=".", print_data=False)
