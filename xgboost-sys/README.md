# xgboost_lib-sys

FFI bindings to [XGBoost](https://xgboost.readthedocs.io/), generated at compile
time with [bindgen](https://github.com/rust-lang-nursery/rust-bindgen).

The bundled XGBoost version is this crate's own version number -- build.rs checks
the two agree against the submodule's `version_config.h`, and release-libs.yml
refuses to publish if they don't. Deliberately not restated here, so there is no
third copy to drift.
