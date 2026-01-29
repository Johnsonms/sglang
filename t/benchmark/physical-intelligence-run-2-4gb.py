#!/usr/bin/env python3
"""
JAX All-Reduce Bandwidth Benchmark - Optimized for 2GB and 4GB buffers
Tests only buffer sizes that work reliably on H100 80GB GPUs
"""

import functools
import logging
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# Setup logging - send to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ],
    force=True
)
logger = logging.getLogger(__name__)


def estimate_all_reduce_bandwidth(
    buffer_size_gb: float = 2.0,
    num_iters: int = 100,
) -> tuple[bool, float, float]:
    """Estimates peak all-reduce bandwidth across all devices using JAX psum.

    Returns:
        (success, algbw_gbps, busbw_gbps) tuple
    """
    try:
        num_devices = jax.device_count()
        num_elements = int(buffer_size_gb * 1024 * 1024 * 1024 / 4)
        buffer_bytes = num_elements * 4

        logger.info(
            f"Running all-reduce bandwidth test: {num_devices} devices, "
            f"{buffer_bytes / 1e9:.2f} GB per device, {num_iters} iterations"
        )

        bus_bw_factor = 2.0 * (num_devices - 1) / num_devices

        devices = np.array(jax.devices())
        mesh = Mesh(devices, axis_names=("dp",))
        dp_spec = PartitionSpec("dp")
        dp_sharding = NamedSharding(mesh, dp_spec)

        @functools.partial(jax.shard_map, mesh=mesh, in_specs=dp_spec, out_specs=dp_spec)
        def single_psum(x: jax.Array) -> jax.Array:
            return jax.lax.psum(x, axis_name="dp")

        def make_all_reduce_loop(n: int):
            @jax.jit
            def all_reduce_loop(data: jax.Array) -> jax.Array:
                def body_fn(_: int, val: jax.Array) -> jax.Array:
                    return single_psum(val)
                return jax.lax.fori_loop(0, n, body_fn, data)
            return all_reduce_loop

        local_devices = jax.local_devices()
        host_array = np.ones((1, num_elements), dtype=np.float32)
        per_device_arrays = [jax.device_put(host_array, device) for device in local_devices]
        data = jax.make_array_from_single_device_arrays((num_devices, num_elements), dp_sharding, per_device_arrays)

        # Correctness check
        try:
            check_result = make_all_reduce_loop(1)(data)
            check_result.block_until_ready()
            expected_value = float(num_devices)
            actual_sample = float(check_result[0, 0])
            if actual_sample != expected_value:
                logger.error(f"Correctness check failed: expected {expected_value}, got {actual_sample}")
                return False, 0.0, 0.0
            logger.info("Correctness check passed")
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
                logger.error(f"Out of memory during correctness check for {buffer_size_gb}GB buffer")
                return False, 0.0, 0.0
            else:
                raise

        # Timed run
        benchmark_fn = make_all_reduce_loop(num_iters)
        benchmark_fn(data).block_until_ready()

        multihost_utils.sync_global_devices("bandwidth_test_start")
        start_time = time.perf_counter()
        benchmark_fn(data).block_until_ready()
        multihost_utils.sync_global_devices("bandwidth_test_end")
        elapsed_sec = time.perf_counter() - start_time

        algbw_gbps = buffer_bytes * num_iters / elapsed_sec / 1e9
        busbw_gbps = algbw_gbps * bus_bw_factor

        logger.info(f"Algorithm bandwidth: {algbw_gbps:.2f} GB/s")
        logger.info(f"Bus bandwidth: {busbw_gbps:.2f} GB/s")

        print(f"\n{'='*70}")
        print(f"✅ RESULT: Algorithm BW = {algbw_gbps:.2f} GB/s, Bus BW = {busbw_gbps:.2f} GB/s")
        print(f"{'='*70}\n")

        return True, algbw_gbps, busbw_gbps

    except Exception as e:
        error_msg = str(e)
        if "RESOURCE_EXHAUSTED" in error_msg or "Out of memory" in error_msg:
            logger.error(f"Out of memory error for {buffer_size_gb}GB buffer")
            logger.info("Try smaller buffer sizes")
        else:
            logger.exception("Error during bandwidth test")
        return False, 0.0, 0.0


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("JAX All-Reduce Bandwidth Benchmark (2GB & 4GB)")
    print("="*70 + "\n")

    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Number of devices: {jax.device_count()}")
    logger.info(f"Device type: {jax.devices()[0].platform}")

    for i, device in enumerate(jax.devices()):
        logger.info(f"Device {i}: {device}")

    print("\n" + "="*70 + "\n")

    # Test only 2GB and 4GB (known to work well)
    buffer_sizes = [2.0, 4.0]
    results = []

    for buffer_size in buffer_sizes:
        logger.info(f"Testing with {buffer_size} GB buffer per device...")
        success, algbw, busbw = estimate_all_reduce_bandwidth(
            buffer_size_gb=buffer_size,
            num_iters=100
        )

        if success:
            results.append({
                'size': buffer_size,
                'algbw': algbw,
                'busbw': busbw
            })

        print("-" * 70 + "\n")

    # Print final summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Successful tests: {len(results)}/{len(buffer_sizes)}")
    print("")

    if results:
        print("Results:")
        for r in results:
            status = "✅" if r['busbw'] >= 460 else "⚠️"
            print(f"  {status} {r['size']:.1f}GB: Alg={r['algbw']:.2f} GB/s, Bus={r['busbw']:.2f} GB/s")

        avg_busbw = sum(r['busbw'] for r in results) / len(results)
        print(f"\nAverage Bus Bandwidth: {avg_busbw:.2f} GB/s")

        if avg_busbw >= 460:
            print("🎯 Target of 460 GB/s EXCEEDED! ✅")
        else:
            print("⚠️  Below 460 GB/s target")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
