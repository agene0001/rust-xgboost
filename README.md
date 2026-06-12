[![Actions Status](https://github.com/agene0001/rust-xgboost/workflows/Macos/badge.svg)](https://github.com/agene0001/rust-xgboost/actions/workflows/macos.yml)
[![Actions Status](https://github.com/agene0001/rust-xgboost/workflows/Linux/badge.svg)](https://github.com/agene0001/rust-xgboost/actions/workflows/linux.yml)
[![Actions Status](https://github.com/agene0001/rust-xgboost/workflows/Windows/badge.svg)](https://github.com/agene0001/rust-xgboost/actions/workflows/windows.yml)


# rust-xgboost


This is mostly a fork of https://github.com/davechallis/rust-xgboost but uses 
another xgboost version and links it dynamically instead of linking it statically as in the original library.

Rust bindings for the [XGBoost](https://xgboost.ai) gradient boosting library.

Creates a shared library and uses Ninja instead of makefiles as generator.

## Requirements

By default the crate builds XGBoost from the pinned submodule (`local_build` feature), so the
headers used for bindgen and the runtime library can never disagree. This requires `cmake`
(and uses `ninja` when available). Alternatively, the `use_prebuilt_xgb` feature downloads an
already compiled library: `--no-default-features --features use_prebuilt_xgb`.

On mac you need to install `libomp` (`brew install libomp`). 
On debian, you need `libclang-dev` (`apt install -y libclang-dev`)

## Documentation

* [Documentation](https://docs.rs/xgboost)

Basic usage example:

```rust
extern crate xgb;

use xgb::{parameters, DMatrix, Booster};

fn main() {
    // training matrix with 5 training examples and 3 features
    let x_train = &[1.0, 1.0, 1.0,
                    1.0, 1.0, 0.0,
                    1.0, 1.0, 1.0,
                    0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0];
    let num_rows = 5;
    let y_train = &[1.0, 1.0, 1.0, 0.0, 1.0];

    // convert training data into XGBoost's matrix format
    let mut dtrain = DMatrix::from_dense(x_train, num_rows).unwrap();

    // set ground truth labels for the training matrix
    dtrain.set_labels(y_train).unwrap();

    // test matrix with 1 row
    let x_test = &[0.7, 0.9, 0.6];
    let num_rows = 1;
    let y_test = &[1.0];
    let mut dtest = DMatrix::from_dense(x_test, num_rows).unwrap();
    dtest.set_labels(y_test).unwrap();

    // configure objectives, metrics, etc.
    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::BinaryLogistic)
        .build().unwrap();

    // configure the tree-based learning model's parameters
    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build().unwrap();

    // overall configuration for Booster
    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(true)
        .build().unwrap();

    // specify datasets to evaluate against during training
    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // overall configuration for training/evaluation
    let params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)                         // dataset to train with
        .boost_rounds(2)                         // number of training iterations
        .booster_params(booster_params)          // model parameters
        .evaluation_sets(Some(evaluation_sets)) // optional datasets to evaluate against in each iteration
        .build().unwrap();

    // train model, and print evaluation data
    let bst = Booster::train(&params).unwrap();

    println!("{:?}", bst.predict(&dtest).unwrap());
}
```

See the [examples](https://github.com/agene0001/rust-xgboost/tree/master/examples) directory for
more detailed examples of different features.

## Performance

For latency-sensitive serving of small batches (roughly under 1000 rows):

* Pin the booster to one thread after loading: `booster.set_param("nthread", "1")`.
  Small-batch latency is dominated by OpenMP thread dispatch; on a 127-feature/50-tree
  binary model this measures ~11-20x faster for single rows.
* Predict straight off your `&[f32]`/CSR slices with `predict_from_dense` /
  `predict_from_csr` (inplace prediction) instead of building a `DMatrix` per request.
* Reuse one output buffer across requests with `predict_from_dense_into` /
  `predict_from_csr_into`. The warm serving loop then performs zero heap allocations
  in the wrapper (verified by `tests/zero_alloc.rs`).

For training and large-batch throughput, two opt-in flags tune the `local_build`
C++ compilation (off by default to keep binaries portable):

```sh
XGB_BUILD_NATIVE=1  # tune codegen for the build machine (-march/-mcpu=native)
XGB_BUILD_IPO=1     # link-time optimization for libxgboost
```

These only apply when building from source and the binary runs on the machine
(or identical CPUs) it was built on. Expect the largest gains from
`XGB_BUILD_NATIVE` on x86-64 hosts with AVX2/AVX-512.

For large training sets with the `hist` tree method, prefer
`DMatrix::from_dense_quantile` / `from_csr_quantile`, which store pre-binned data
(~1 byte per value instead of 4) and skip a separate sketching pass.

## Status

The version number tracks the bundled XGBoost version.

This is still a very early stage of development, so the API is changing as usability issues occur,
or new features are supported. This is still expected to be compatible to an earlier rust-xgboost library.

Builds against XGBoost 3.2.0.

## Use prebuilt xgboost library or build it

Xgboost is kind of complicated to compile, especially when there is GPU support involved.
It is sometimes easier to use a pre-build library. Therefore, the feature flag `use_prebuilt_xgb` is enabled by default.
This is using a prebuilt shared library in xboost-sys/lib by default. You can also use a custom folder by defining `$XGBOOST_LIB_DIR`.

If you prefer to use xgboost from homebrew, which may have GPU support, your can for example define
```
XGBOOST_LIB_DIR=${HOMEBREW_PREFIX}/opt/xgboost/lib
```

If you want to use it by yourself, you can disable the use_prebuild_xgb feature:
```
xgb = { version = "3",  default-features = false, features=["local_build"] }
```
This would require `cmake` and `ninja-build` as build dependencies.

If you want build it locally, after cloning, perform `git submodule update --init --recursive`
to install submodule dependencies.

brew commands for MacOs to compile locally:
- brew install libomp
- brew install cmake
- brew install ninja
- brew install llvm

### Supported Platforms

Prebuilt lib and built locally:

* Mac OS
* Linux

Prebuilt lib only

* Windows 

Local windows built is possible, but steps may require manual copy of VS output files.

GPU support on windows:

How to get a .lib and .dll from pip , using a VS Developer CMD prompt:
```
python3 -m venv .venv
.venv\Scripts\activate.bat
pip install xgboost
pip show xgboost
# check Location entry
copy {Location}\xgboost.dll .
gendef xgboost.dll
lib /def:xgboost.def /machine:x64" /out:xgboost.lib
```
