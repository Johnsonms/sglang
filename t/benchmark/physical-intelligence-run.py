def estimate_all_reduce_bandwidth(
    buffer_size_gb: float = 2.0,
    num_iters: int = 100,
) -> bool:
    """Estimates peak all-reduce bandwidth across all devices using JAX psum.

    Runs a large all-reduce operation to maximize and measure bandwidth.
    Uses shard_map with a mesh for proper per-device data allocation.

    Args:
        buffer_size_gb: Buffer size in GB per device.
        num_iters: Number of all-reduce iterations for the timed run.
        warmup_iters: Number of all-reduce iterations for warmup/compilation.

    Returns:
        True if the benchmark completes successfully, False otherwise.
    """
    try:
        num_devices = jax.device_count()
        num_elements = int(buffer_size_gb * 1024 * 1024 * 1024 / 4)  # float32 = 4 bytes
        buffer_bytes = num_elements * 4

        logger.info(
            f"Running all-reduce bandwidth test: {num_devices} devices, "
            f"{buffer_bytes / 1e9:.2f} GB per device, {num_iters} iterations"
        )

        # For all-reduce, bus bandwidth = algbw * 2 * (n-1) / n
        bus_bw_factor = 2.0 * (num_devices - 1) / num_devices

        # Create mesh with all devices for data parallelism
        devices = np.array(jax.devices())
        mesh = sharding.Mesh(devices, axis_names=("dp",))
        dp_spec = sharding.PartitionSpec("dp")
        dp_sharding = sharding.NamedSharding(mesh, dp_spec)

        @functools.partial(jax.shard_map, mesh=mesh, in_specs=dp_spec, out_specs=dp_spec)
        def single_psum(x: jax.Array) -> jax.Array:
            return jax.lax.psum(x, axis_name="dp")

        def make_all_reduce_loop(n: int):
            """Create a jitted function that runs n all-reduce iterations using fori_loop."""

            @jax.jit
            def all_reduce_loop(data: jax.Array) -> jax.Array:
                def body_fn(_: int, val: jax.Array) -> jax.Array:
                    return single_psum(val)

                return jax.lax.fori_loop(0, n, body_fn, data)

            return all_reduce_loop

        # Create per-device buffers on host, then place directly on each device.
        # This avoids allocating a monolithic array on device 0.
        # Each per-device array has shape (1, num_elements) to match sharding expectation.
        local_devices = jax.local_devices()
        host_array = np.ones((1, num_elements), dtype=np.float32)
        per_device_arrays = [jax.device_put(host_array, device) for device in local_devices]
        data = jax.make_array_from_single_device_arrays((num_devices, num_elements), dp_sharding, per_device_arrays)

        # Correctness check: single psum iteration on all-ones should give num_devices
        check_result = make_all_reduce_loop(1)(data)
        check_result.block_until_ready()
        expected_value = float(num_devices)
        actual_sample = float(check_result[0, 0])
        if actual_sample != expected_value:
            logger.error(f"Correctness check failed: expected {expected_value}, got {actual_sample}")
            return False
        logger.info("Correctness check passed")

        # Timed run - compile first, then time
        benchmark_fn = make_all_reduce_loop(num_iters)
        benchmark_fn(data).block_until_ready()  # Compile

        multihost_utils.sync_global_devices("bandwidth_test_start")
        start_time = time.perf_counter()
        benchmark_fn(data).block_until_ready()
        multihost_utils.sync_global_devices("bandwidth_test_end")
        elapsed_sec = time.perf_counter() - start_time

        algbw_gbps = buffer_bytes * num_iters / elapsed_sec / 1e9
        busbw_gbps = algbw_gbps * bus_bw_factor

        logger.info(f"Algorithm bandwidth: {algbw_gbps:.2f} GB/s")
        logger.info(f"Bus bandwidth: {busbw_gbps:.2f} GB/s")

        return True

    except Exception:
        logger.exception("Error during bandwidth test, check NCCL/NVLink/InfiniBand status.")
        return False
