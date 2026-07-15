# Optimizing services that use the `xgb` crate

The wrapper itself is already allocation-free on hot paths. The remaining
performance is unlocked by **how you build** the crate and **which APIs you
call**. Apply everything below that matches your workload.

Benchmark figures quoted here were measured on an Apple M3 Pro with a
127-feature/50-tree binary model (`benches/` in this repo). The *direction* of
each effect is universal; exact crossover points vary by hardware, so re-verify
them on your deployment machine if they matter to you.

## 1. Build-time: native codegen for the bundled C++

All prediction/training time is spent inside libxgboost (C++), which the crate
compiles from source by default (`local_build` feature). Two opt-in env vars
tune that build:

- `XGB_BUILD_NATIVE=1` — compiles libxgboost with `-march=native` (x86-64) /
  `-mcpu=native` (aarch64). Biggest gains on x86-64 hosts with AVX2/AVX-512.
- `XGB_BUILD_IPO=1` — enables link-time optimization (CMake IPO) for libxgboost.

**Only use these if the binary runs on the machine (or identical CPU family) it
was built on.** They are off by default so release binaries stay portable — a
`-march=native` build from a newer CPU will crash with an illegal-instruction
fault on older ones.

**Critical operational detail:** the build script watches these env vars
(`rerun-if-env-changed`). If you set them on one `cargo` command and not the
next, cargo silently rebuilds the whole C++ library back to the default
configuration (slow, and you lose the tuned build without noticing). Don't set
them ad hoc in the shell — pin them in the project's `.cargo/config.toml` so
every cargo invocation agrees:

```toml
[env]
XGB_BUILD_NATIVE = "1"
XGB_BUILD_IPO = "1"
```

**Transitive dependency chains:** these env vars must be pinned in the
**final binary's** workspace — the project where `cargo build` actually runs.
`.cargo/config.toml` does not travel with a dependency: if a library that
wraps this crate pins the vars in its own repo, that works for the library's
tests and benches but is silently ignored when a higher-level project builds
the library as a dependency (that project gets a default portable libxgboost).
Every deployable at the top of the chain needs the `[env]` block above in its
own `.cargo/config.toml` or CI environment.

## 2. Serving small batches (< ~1000 rows per call)

### a. Pin the booster to one thread after loading

```rust
let mut booster = Booster::load_buffer(&model_bytes)?;
booster.set_param("nthread", "1")?;
```

Small-batch latency is dominated by OpenMP thread dispatch, not tree traversal.
Measured: `nthread=1` is **~11x faster for 1 row**, ~5x for 16 rows, ~2x for
100 rows. Multithreading only wins again above roughly 1000 rows per call. If
you serve both tiny and huge batches, keep two boosters loaded from the same
bytes with different `nthread` settings.

### b. Predict straight off your slices — never build a `DMatrix` per request

```rust
// dense: row-major &[f32]
let (preds, shape) = booster.predict_from_dense(&features, num_rows)?;

// sparse CSR
let (preds, shape) = booster.predict_from_csr(&indptr, &indices, &data, num_cols)?;
```

Constructing a `DMatrix` per request is pure overhead (it copies and re-indexes
the data) and dominates end-to-end latency at small batch sizes. The
`predict_from_*` (inplace) paths skip it entirely.

### c. Reuse one output buffer in the request loop

```rust
let mut out: Vec<f32> = Vec::new();
loop {
    booster.predict_from_dense_into(&features, num_rows, &mut out)?;
    // out.len() == num_rows * n_groups  (n_groups = 1 for regression/binary)
}
```

`predict_from_dense_into` / `predict_from_csr_into` / `predict_into` (the
`DMatrix`-path variant, for batch scoring) write into a caller-owned `Vec`
instead of allocating. Once warm, the serving loop performs **zero heap
allocations** in the wrapper — asserted by this repo's test suite
(`tests/zero_alloc.rs`), so it's a guarantee, not a hope.

### d. Threading model for servers

`Booster` and `DMatrix` are `Send` but **not** `Sync`: you can load a model
once and move it into a worker thread (thread pool, tokio task), but you cannot
share one instance behind `&`/`Arc` across concurrent callers — concurrent
predictions would race on internal cached state. The intended patterns:

- **One booster per worker thread** (preferred): load the model bytes once,
  call `Booster::load_buffer` per thread. Loading is cheap relative to holding
  a lock on the hot path.
- Or a single owning thread fed by a channel, if you must have exactly one
  instance.

Avoid `Arc<Mutex<Booster>>` for high-QPS serving — it serializes all
predictions.

## 3. Training

- **Use `QuantileDMatrix` for large in-memory training sets with the `hist`
  tree method** (the default):
  ```rust
  let dtrain = DMatrix::from_dense_quantile(&data, num_rows, Some(&labels), 256)?;
  ```
  It stores pre-binned values (~1 byte per feature value instead of 4 — roughly
  **4x less memory**) and skips a separate sketching pass at the start of
  training. The last argument (`max_bin`) **must match** the booster's
  `max_bin` parameter (default 256) or training errors out. Sparse counterpart:
  `from_csr_quantile`. Note: quantile matrices are training-only — they cannot
  be saved with `DMatrix::save`.
- After training, if you keep the booster in the same process for inference,
  call `booster.reset()` to free XGBoost's internal training caches (gradient
  buffers, prediction caches) without touching the model.

## 4. If you are a library wrapping this crate

The API-level optimizations above (thread pinning, inplace prediction, buffer
reuse) live in *your* code, and your consumers inherit them automatically — a
higher-level project calling your API needs no `xgb`-specific changes. Two
things do cross the boundary to your consumers; state them in your docs:

- **Build flags don't propagate** (see the transitive-chain note in section 1):
  the final binary's workspace must pin `XGB_BUILD_NATIVE`/`XGB_BUILD_IPO`
  itself, or it gets a default portable libxgboost.
- **Document your tuning regime**: say which batch sizes and threading model
  you optimized for (e.g. "single-row/small-batch, one booster per worker
  thread, `nthread=1`"). If a consumer funnels 100k-row batches through an API
  tuned for single rows (or vice versa), the `nthread` choice inverts — above
  roughly 1000 rows per call, multithreaded prediction wins again.

## 5. Data-format thresholds (already automatic — don't work around them)

`DMatrix::from_dense` and `from_csr`/`from_csc` internally auto-select single-
vs multi-threaded construction based on empirically benchmarked crossover
points (50k elements dense, 30k non-zeros sparse). Don't pre-chunk or pad data
to influence this; just call them directly.
