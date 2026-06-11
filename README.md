[![Actions Status](https://github.com/agene0001/rust-xgboost/workflows/Macos/badge.svg)](https://github.com/agene0001/rust-xgboost/actions/workflows/macos.yml)
[![Actions Status](https://github.com/agene0001/rust-xgboost/workflows/Linux/badge.svg)](https://github.com/agene0001/rust-xgboost/actions/workflows/linux.yml)
[![Actions Status](https://github.com/agene0001/rust-xgboost/workflows/Windows/badge.svg)](https://github.com/agene0001/rust-xgboost/actions/workflows/windows.yml)


# rust-xgboost


This is mostly a fork of https://github.com/davechallis/rust-xgboost but uses 
another xgboost version and links it dynamically instead of linking it statically as in the original library.

Rust bindings for the [XGBoost](https://xgboost.ai) gradient boosting library.

Creates a shared library and uses Ninja instead of makefiles as generator.

## Requirements

It is highly recommended to use the `use_prebuilt_xgb` feature, which is enabled by default.
It will use an already compiled xgboost library which will be downloaded as build step of this crate.
On Mac, it will use an arm64 shared library. On windows and linux, it is using x64 architecture.

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

## Status

The version number is just an indicator that xboost 3.0.0 is used.

This is still a very early stage of development, so the API is changing as usability issues occur,
or new features are supported. This is still expected to be compatible to an earlier rust-xgboost library.

Builds against XGBoost 3.0.0.

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

### Running binaries outside `cargo run`

libxgboost is linked dynamically. `cargo run` and `cargo test` always work
because Cargo adds the library's directory to the loader's environment
(`PATH` / `LD_LIBRARY_PATH` / `DYLD_FALLBACK_LIBRARY_PATH`) for the child
process — but a binary started directly (e.g. `./target/release/myapp`) gets
no such help.

To support that, the build script stages the shared library
(`xgboost.dll` / `libxgboost.so` / `libxgboost.dylib`) next to the
executables in the target profile directory, for both the `local_build` and
`use_prebuilt_xgb` paths.

On **Windows** that is sufficient — the loader searches the exe's directory.

On **Linux/macOS** the loader only looks next to the exe if the binary
carries an `$ORIGIN` / `@loader_path` rpath. Cargo does not propagate linker
args from dependency build scripts, so the *binary* crate has to add it
itself. Either in the binary crate's `build.rs`:

```rust
fn main() {
    let target = std::env::var("TARGET").unwrap();
    if target.contains("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    } else if target.contains("apple") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    }
}
```

or in its `.cargo/config.toml`:

```toml
[target.'cfg(target_os = "linux")']
rustflags = ["-C", "link-arg=-Wl,-rpath,$ORIGIN"]

[target.'cfg(target_os = "macos")']
rustflags = ["-C", "link-arg=-Wl,-rpath,@loader_path"]
```

When deploying, copy the staged library alongside the executable.

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
