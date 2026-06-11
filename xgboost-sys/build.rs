use bindgen;
use std::env;
use std::path::{Path, PathBuf};

/// Primary source of prebuilt binaries: GitHub release assets built from the
/// pinned submodule by `.github/workflows/release-libs.yml` (flat asset names,
/// `<target>-<file>`). Bump the tag here together with the submodule.
#[cfg(feature = "use_prebuilt_xgb")]
const PREBUILT_RELEASE_URL: &str = "https://github.com/agene0001/rust-xgboost/releases/download/v3.2.0";

/// Fallback: XGBoost 3.0.0 binaries committed in the upstream fork's repo at its
/// v3.0.1 tag. These are OLDER than the bundled headers (version skew), so a
/// `cargo:warning` is emitted whenever they are used.
#[cfg(feature = "use_prebuilt_xgb")]
const LEGACY_URL: &str = "https://github.com/marcomq/rust-xgboost/raw/refs/tags/v3.0.1/xgboost-sys/lib";

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    // dunce strips Windows' \\?\ extended-length prefix, which std's
    // canonicalize produces and which breaks CMake's file(GLOB) (configure
    // fails with "No SOURCES given to target: xgboost").
    let xgb_root = dunce::canonicalize(Path::new("xgboost")).unwrap();

    let wrapper_h = xgb_root.join("include").join("xgboost").join("c_api.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper_h.to_string_lossy())
        .clang_arg(format!("-I{}", xgb_root.join("include").display()))
        .clang_arg(format!("-I{}", xgb_root.join("dmlc-core").join("include").display()));

    #[cfg(feature = "cuda")]
    let bindings = bindings.clang_arg("-I/usr/local/cuda/include");
    let bindings = bindings.generate().expect("Unable to generate bindings.");

    let out_path = PathBuf::from(&out_dir);
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    if target.contains("apple") {
        println!(
            "cargo:rustc-link-search=native={}/opt/libomp/lib",
            &std::env::var("HOMEBREW_PREFIX").unwrap_or("/opt/homebrew".into())
        );
    }

    #[cfg(feature = "use_prebuilt_xgb")]
    {
        if let Ok(xgboost_lib_dir) = std::env::var("XGBOOST_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
        } else {
            let deps_path = dunce::canonicalize(Path::new(&format!("{}/../../../deps", out_dir))).unwrap();
            let deps_path = deps_path.to_string_lossy();
            println!("cargo:rustc-link-search=native={}", deps_path);
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                if !std::fs::exists(format!("{deps_path}/libxgboost.dylib")).unwrap() {
                    fetch_lib("mac_arm64", "libxgboost.dylib", &deps_path).unwrap();
                    fetch_lib("mac_arm64", "libdmlc.a", &deps_path).unwrap();
                }
            } else if cfg!(target_os = "linux") {
                let target_dir = if cfg!(target_arch = "aarch64") {
                    "linux_arm64"
                } else {
                    "linux_amd64"
                };
                if !std::fs::exists(format!("{deps_path}/libxgboost.so")).unwrap() {
                    fetch_lib(target_dir, "libxgboost.so", &deps_path).unwrap();
                    fetch_lib(target_dir, "libdmlc.a", &deps_path).unwrap();
                }
            } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
                if !std::fs::exists(format!("{deps_path}/xgboost.dll")).unwrap() {
                    fetch_lib("win_amd64", "xgboost.dll", &deps_path).unwrap();
                    fetch_lib("win_amd64", "xgboost.lib", &deps_path).unwrap();
                }
                stage_dll_next_to_exe(Path::new(&format!("{deps_path}/xgboost.dll")), &out_dir);
            } else {
                if let Ok(homebrew_path) = std::env::var("HOMEBREW_PREFIX") {
                    let xgboost_lib_dir = format!("{}/opt/xgboost/lib", &homebrew_path);
                    println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
                } else {
                    panic!("Please set $XGBOOST_LIB_DIR")
                }
            }
        }
    }

    #[cfg(feature = "local_build")]
    {
        // compile XGBOOST with cmake (and ninja, when available)

        // CMake
        let mut dst = cmake::Config::new(&xgb_root);
        let ninja_available = std::process::Command::new("ninja")
            .arg("--version")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false);
        let dst = if ninja_available { dst.generator("Ninja") } else { &mut dst };
        // Release (-O3, no debug info) to match upstream's distributed builds;
        // RelWithDebInfo's -O2 leaves performance on the table in the hot loops.
        let dst = dst.define("CMAKE_BUILD_TYPE", "Release");

        #[cfg(feature = "cuda")]
        let mut dst = dst
            .define("USE_CUDA", "ON")
            .define("BUILD_WITH_CUDA", "ON")
            .define("BUILD_WITH_CUDA_CUB", "ON");

        let dst = dst.build();

        println!("cargo:rustc-link-search=native={}", dst.display());
        println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
        println!("cargo:rustc-link-search=native={}", dst.join("lib64").display());
        println!("cargo:rustc-link-lib=static=dmlc");

        if cfg!(target_os = "windows") {
            stage_dll_next_to_exe(&dst.join("bin").join("xgboost.dll"), &out_dir);
        }
    }

    // link to appropriate C++ lib
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=omp");
    } else {
        #[cfg(target_os = "linux")]
        {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=stdc++fs");
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }

    println!("cargo:rustc-link-lib=dylib=xgboost");

    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-search={}", "/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=static=cudart_static");
    }
}

/// Windows' loader only searches the exe's directory, system dirs, and PATH —
/// not the cmake out dir or target/<profile>/deps where the DLL lands, so a
/// binary run outside `cargo run` (which patches PATH) fails with
/// "xgboost.dll was not found". Stage a copy in the profile root, where the
/// final executables are emitted.
fn stage_dll_next_to_exe(dll_src: &Path, out_dir: &str) {
    // OUT_DIR = target/<profile>/build/<crate>-<hash>/out → ../../.. = profile root
    let profile_dir = match dunce::canonicalize(Path::new(&format!("{}/../../..", out_dir))) {
        Ok(dir) => dir,
        Err(e) => {
            println!("cargo:warning=could not resolve profile dir to stage xgboost.dll: {e}");
            return;
        }
    };
    if let Err(e) = std::fs::copy(dll_src, profile_dir.join("xgboost.dll")) {
        // Non-fatal: a running executable can hold a lock on the old copy.
        println!(
            "cargo:warning=could not stage {} into {}: {e}",
            dll_src.display(),
            profile_dir.display()
        );
    }
}

#[cfg(feature = "use_prebuilt_xgb")]
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Download one prebuilt library into `deps_path`, preferring this fork's
/// release assets and falling back to the legacy (older) binaries with a
/// warning when the release asset is unavailable.
#[cfg(feature = "use_prebuilt_xgb")]
fn fetch_lib(target_dir: &str, file: &str, deps_path: &str) -> Result<()> {
    let dest = format!("{deps_path}/{file}");
    let primary = format!("{PREBUILT_RELEASE_URL}/{target_dir}-{file}");
    if web_copy(&primary, &dest).is_ok() {
        return Ok(());
    }
    println!(
        "cargo:warning=prebuilt asset {primary} unavailable; falling back to legacy XGBoost 3.0.0 \
         binaries, which are older than the bundled headers. Publish release assets via the \
         release-libs workflow, or build from source with the local_build feature."
    );
    web_copy(&format!("{LEGACY_URL}/{target_dir}/{file}"), &dest)
}

#[cfg(feature = "use_prebuilt_xgb")]
fn web_copy(web_src: &str, target: &str) -> Result<()> {
    dbg!(&web_src);
    // error_for_status is load-bearing: without it a 404 page would be written
    // out as the library file, and the fallback in fetch_lib would never run.
    let resp = reqwest::blocking::get(web_src)?.error_for_status()?;
    let body = resp.bytes()?;
    std::fs::write(target, &body)?;
    Ok(())
}
