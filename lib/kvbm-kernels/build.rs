// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Declare the stub_kernels cfg so Rust knows it's a valid cfg option
    println!("cargo:rustc-check-cfg=cfg(stub_kernels)");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();

    // Track CUDA file changes
    let cu_files = discover_cuda_files();
    for file in &cu_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }

    // Track HIP file changes
    let hip_files = discover_hip_files();
    for file in &hip_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }

    println!(
        "cargo:rerun-if-changed={}",
        Path::new(&manifest_dir).join("cuda/stubs.c").display()
    );
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_PTX_ARCHS");
    println!("cargo:rerun-if-env-changed=KVBM_REQUIRE_CUDA");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=KVBM_REQUIRE_ROCM");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_ARCHS");

    // Check if CUDA or ROCm is required (set by Python bindings build)
    let require_cuda = env::var("KVBM_REQUIRE_CUDA").is_ok();
    let require_rocm = env::var("KVBM_REQUIRE_ROCM").is_ok();
    let nvcc_available = is_nvcc_available();
    let hipcc_available = is_hipcc_available();

    // Fail early if CUDA required but not available
    if require_cuda && !nvcc_available {
        panic!(
            "\n\n\
            ╔════════════════════════════════════════════════════════════════════════╗\n\
            ║  KVBM_REQUIRE_CUDA is set but nvcc is not available!                   ║\n\
            ║                                                                        ║\n\
            ║  Python bindings require real CUDA kernels. Please:                    ║\n\
            ║    1. Install CUDA toolkit with nvcc, or                               ║\n\
            ║    2. Unset KVBM_REQUIRE_CUDA for stub-only build                      ║\n\
            ╚════════════════════════════════════════════════════════════════════════╝\n\
            "
        );
    }

    // Fail early if ROCm required but not available
    if require_rocm && !hipcc_available {
        panic!(
            "\n\n\
            ╔════════════════════════════════════════════════════════════════════════╗\n\
            ║  KVBM_REQUIRE_ROCM is set but hipcc is not available!                  ║\n\
            ║                                                                        ║\n\
            ║  Python bindings require real HIP kernels. Please:                     ║\n\
            ║    1. Install ROCm toolkit with hipcc, or                              ║\n\
            ║    2. Unset KVBM_REQUIRE_ROCM for stub-only build                      ║\n\
            ╚════════════════════════════════════════════════════════════════════════╝\n\
            "
        );
    }

    // Determine build mode
    let build_mode = determine_build_mode(nvcc_available, hipcc_available);

    // Check if static linking is requested (only applies to CUDA builds, not stubs)
    #[cfg(feature = "static-kernels")]
    let use_static = true;
    #[cfg(not(feature = "static-kernels"))]
    let use_static = false;

    match build_mode {
        BuildMode::FromSource => {
            if use_static {
                println!("cargo:warning=Building CUDA kernels from source (static linking)");
            } else {
                println!("cargo:warning=Building CUDA kernels from source (dynamic linking)");
            }
            build_cuda_library(&cu_files, &out_dir, use_static);
        }
        BuildMode::FromSourceHip => {
            if use_static {
                println!("cargo:warning=Building HIP kernels from source (static linking)");
            } else {
                println!("cargo:warning=Building HIP kernels from source (dynamic linking)");
            }
            build_hip_library(&hip_files, &out_dir, use_static);
        }
        BuildMode::Stubs => {
            println!("cargo:warning=Building stub kernels (no CUDA/ROCm available, dynamic linking)");
            build_stub_shared_library(&manifest_dir, &out_dir);
            println!("cargo:rustc-cfg=stub_kernels");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BuildMode {
    FromSource,
    FromSourceHip,
    Stubs,
}

/// Determine the build mode based on compiler availability.
/// Prefers nvcc (CUDA) over hipcc (ROCm) when both are present.
fn determine_build_mode(nvcc_available: bool, hipcc_available: bool) -> BuildMode {
    if nvcc_available {
        BuildMode::FromSource
    } else if hipcc_available {
        BuildMode::FromSourceHip
    } else {
        BuildMode::Stubs
    }
}

fn is_nvcc_available() -> bool {
    Command::new("nvcc").arg("--version").output().is_ok()
}

fn is_hipcc_available() -> bool {
    Command::new("hipcc").arg("--version").output().is_ok()
        || Command::new("/opt/rocm/bin/hipcc")
            .arg("--version")
            .output()
            .is_ok()
}

/// Resolve the hipcc binary path, checking ROCM_PATH/HIP_PATH env vars,
/// then PATH, then the standard /opt/rocm location.
fn resolve_hipcc() -> String {
    if let Ok(hip_path) = env::var("HIP_PATH") {
        let candidate = format!("{}/bin/hipcc", hip_path);
        if Path::new(&candidate).exists() {
            return candidate;
        }
    }
    if let Ok(rocm_path) = env::var("ROCM_PATH") {
        let candidate = format!("{}/bin/hipcc", rocm_path);
        if Path::new(&candidate).exists() {
            return candidate;
        }
    }
    if Command::new("hipcc").arg("--version").output().is_ok() {
        return "hipcc".to_string();
    }
    "/opt/rocm/bin/hipcc".to_string()
}

/// Build CUDA kernels from source.
/// If `use_static` is true, builds a static archive (.a); otherwise builds a shared library (.so).
fn build_cuda_library(cu_files: &[PathBuf], out_dir: &str, use_static: bool) {
    let arch_flags = get_cuda_arch_flags();

    // Only build tensor_kernels.cu into the library (it has the extern "C" functions)
    let tensor_kernels_path = cu_files
        .iter()
        .find(|p| p.file_stem().unwrap() == "tensor_kernels")
        .expect("tensor_kernels.cu not found");

    let obj_path = Path::new(out_dir).join("kvbm_kernels.o");

    // Step 1: Compile to object file
    let mut nvcc_cmd = Command::new("nvcc");
    nvcc_cmd
        .arg("-m64")
        .arg("-c")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg(tensor_kernels_path)
        .arg("-o")
        .arg(&obj_path);

    for flag in &arch_flags {
        nvcc_cmd.arg(flag);
    }

    println!("cargo:warning=Compiling tensor_kernels.cu to object file...");
    let status = nvcc_cmd
        .status()
        .expect("Failed to execute nvcc for object file");

    if !status.success() {
        panic!("nvcc failed to compile tensor_kernels.cu");
    }

    if use_static {
        // Step 2a: Create static archive
        let ar_path = Path::new(out_dir).join("libkvbm_kernels.a");

        let mut ar_cmd = Command::new("ar");
        ar_cmd.arg("crus").arg(&ar_path).arg(&obj_path);

        println!("cargo:warning=Creating static archive libkvbm_kernels.a...");
        let status = ar_cmd
            .status()
            .expect("Failed to execute ar for static archive");

        if !status.success() {
            panic!("ar failed to create static archive");
        }

        // Set up static linking
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=kvbm_kernels");

        // Add CUDA runtime library paths and link cudart dynamically
        add_cuda_library_paths();
        println!("cargo:rustc-link-lib=cudart");

        // CUDA object code compiled by nvcc contains C++ runtime symbols
        // (operator new/delete, __gxx_personality_v0, etc.)
        println!("cargo:rustc-link-lib=stdc++");
    } else {
        // Step 2b: Link into shared library
        let so_path = Path::new(out_dir).join("libkvbm_kernels.so");

        let mut link_cmd = Command::new("nvcc");
        link_cmd
            .arg("-shared")
            .arg("-o")
            .arg(&so_path)
            .arg(&obj_path)
            .arg("-lcudart");

        println!("cargo:warning=Linking kvbm_kernels into shared library...");
        let status = link_cmd
            .status()
            .expect("Failed to execute nvcc for linking");

        if !status.success() {
            panic!("nvcc failed to link shared library");
        }

        // Set up dynamic linking
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=dylib=kvbm_kernels");

        // Add CUDA runtime library paths
        add_cuda_library_paths();
        println!("cargo:rustc-link-lib=cudart");
    }
}

/// Build stub shared library from stubs.c when CUDA is not available.
fn build_stub_shared_library(manifest_dir: &str, out_dir: &str) {
    let stubs_path = Path::new(manifest_dir).join("cuda/stubs.c");

    if !stubs_path.exists() {
        panic!(
            "Stub source file not found at {}. Cannot build without CUDA.",
            stubs_path.display()
        );
    }

    // Build shared library from stubs.c using the system C compiler
    let so_path = Path::new(out_dir).join("libkvbm_kernels.so");
    let obj_path = Path::new(out_dir).join("stubs.o");

    // Compile to object file
    let mut gcc_compile = Command::new("cc");
    gcc_compile
        .arg("-c")
        .arg("-fPIC")
        .arg("-O2")
        .arg(&stubs_path)
        .arg("-o")
        .arg(&obj_path);

    println!("cargo:warning=Compiling stubs.c...");
    let status = gcc_compile
        .status()
        .expect("Failed to execute cc for stubs");

    if !status.success() {
        panic!("Failed to compile stubs.c");
    }

    // Link into shared library
    let mut gcc_link = Command::new("cc");
    gcc_link
        .arg("-shared")
        .arg("-o")
        .arg(&so_path)
        .arg(&obj_path);

    println!("cargo:warning=Linking stub shared library...");
    let status = gcc_link.status().expect("Failed to link stub library");

    if !status.success() {
        panic!("Failed to link stub shared library");
    }

    // Set up linking
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=dylib=kvbm_kernels");
}

/// Build HIP kernels from source using hipcc.
/// If `use_static` is true, builds a static archive (.a); otherwise builds a shared library (.so).
fn build_hip_library(hip_files: &[PathBuf], out_dir: &str, use_static: bool) {
    let arch_flags = get_rocm_arch_flags();
    let hipcc = resolve_hipcc();

    let tensor_kernels_path = hip_files
        .iter()
        .find(|p| p.file_stem().unwrap() == "tensor_kernels")
        .expect("tensor_kernels.hip not found");

    let obj_path = Path::new(out_dir).join("kvbm_kernels.o");

    let mut hipcc_cmd = Command::new(&hipcc);
    hipcc_cmd
        .arg("-c")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-fPIC")
        .arg(tensor_kernels_path)
        .arg("-o")
        .arg(&obj_path);

    for flag in &arch_flags {
        hipcc_cmd.arg(flag);
    }

    println!("cargo:warning=Compiling tensor_kernels.hip to object file with {}...", hipcc);
    let status = hipcc_cmd
        .status()
        .expect("Failed to execute hipcc for object file");

    if !status.success() {
        panic!("hipcc failed to compile tensor_kernels.hip");
    }

    if use_static {
        let ar_path = Path::new(out_dir).join("libkvbm_kernels.a");

        let mut ar_cmd = Command::new("ar");
        ar_cmd.arg("crus").arg(&ar_path).arg(&obj_path);

        println!("cargo:warning=Creating static archive libkvbm_kernels.a...");
        let status = ar_cmd
            .status()
            .expect("Failed to execute ar for static archive");

        if !status.success() {
            panic!("ar failed to create static archive");
        }

        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=kvbm_kernels");

        add_rocm_library_paths();
        println!("cargo:rustc-link-lib=amdhip64");
        println!("cargo:rustc-link-lib=stdc++");
    } else {
        let so_path = Path::new(out_dir).join("libkvbm_kernels.so");

        let mut link_cmd = Command::new(&hipcc);
        link_cmd
            .arg("-shared")
            .arg("-o")
            .arg(&so_path)
            .arg(&obj_path)
            .arg("-lamdhip64");

        println!("cargo:warning=Linking kvbm_kernels into shared library (HIP)...");
        let status = link_cmd
            .status()
            .expect("Failed to execute hipcc for linking");

        if !status.success() {
            panic!("hipcc failed to link shared library");
        }

        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=dylib=kvbm_kernels");

        add_rocm_library_paths();
        println!("cargo:rustc-link-lib=amdhip64");
    }
}

fn discover_cuda_files() -> Vec<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let cuda_dir = Path::new(&manifest_dir).join("cuda");
    let mut cu_files = Vec::new();

    for entry in fs::read_dir(cuda_dir).expect("Failed to read cuda directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "cu") {
            cu_files.push(path);
        }
    }
    cu_files
}

fn discover_hip_files() -> Vec<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let hip_dir = Path::new(&manifest_dir).join("hip");
    let mut hip_files = Vec::new();

    if !hip_dir.exists() {
        return hip_files;
    }

    for entry in fs::read_dir(hip_dir).expect("Failed to read hip directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "hip") {
            hip_files.push(path);
        }
    }
    hip_files
}

/// Return hipcc `--offload-arch=` flags for the target GPU architectures.
/// Defaults to gfx942 (MI300X) and gfx950 (MI355X) when ROCM_ARCHS is unset.
fn get_rocm_arch_flags() -> Vec<String> {
    let explicit_archs = env::var("ROCM_ARCHS").ok();
    let arch_list = explicit_archs.as_deref().unwrap_or("gfx942,gfx950");

    let mut flags = Vec::new();
    for arch in arch_list.split(',') {
        let arch = arch.trim();
        if arch.is_empty() {
            continue;
        }
        println!("cargo:warning=Including ROCm target: {}", arch);
        flags.push(format!("--offload-arch={}", arch));
    }
    flags
}

fn add_rocm_library_paths() {
    if let Ok(hip_path) = env::var("HIP_PATH") {
        println!("cargo:rustc-link-search=native={}/lib", hip_path);
        println!("cargo:rustc-link-search=native={}/lib64", hip_path);
    } else if let Ok(rocm_path) = env::var("ROCM_PATH") {
        println!("cargo:rustc-link-search=native={}/lib", rocm_path);
        println!("cargo:rustc-link-search=native={}/lib64", rocm_path);
    } else {
        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib64");
    }
}

/// Parse CUDA toolkit version from `nvcc --version` output.
/// Returns (major, minor) tuple, e.g. (12, 8) for CUDA 12.8.
fn parse_cuda_version() -> Option<(u32, u32)> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    // nvcc output contains a line like: "Cuda compilation tools, release 12.8, V12.8.89"
    for line in stdout.lines() {
        if let Some(pos) = line.find("release ") {
            let after = &line[pos + "release ".len()..];
            let version_str = after.split(',').next().unwrap_or("").trim();
            let mut parts = version_str.split('.');
            let major = parts.next()?.parse::<u32>().ok()?;
            let minor = parts.next()?.parse::<u32>().ok()?;
            return Some((major, minor));
        }
    }
    None
}

/// Return the maximum supported compute capability for a given CUDA toolkit version.
///
/// Panics if the CUDA version is below 12.0.
fn max_supported_compute(cuda_version: (u32, u32)) -> u32 {
    match cuda_version {
        (major, _) if major < 12 => {
            panic!("CUDA {major}.x is not supported; CUDA 12.0 or newer is required")
        }
        (12, minor) if minor >= 8 => 120,
        (major, _) if major >= 13 => 120,
        _ => 90,
    }
}

fn get_cuda_arch_flags() -> Vec<String> {
    let mut flags = Vec::new();

    let cuda_version = parse_cuda_version();
    let max_compute = cuda_version.map(max_supported_compute);

    if let Some((major, minor)) = cuda_version {
        println!(
            "cargo:warning=Detected CUDA {}.{}, max supported compute: sm_{}",
            major,
            minor,
            max_compute.unwrap()
        );
    } else {
        println!("cargo:warning=Could not detect CUDA version, including all architectures");
    }

    let explicit_archs = env::var("CUDA_ARCHS").ok();
    let arch_list = explicit_archs.as_deref().unwrap_or("80,86,89,90,100,120");

    for arch in arch_list.split(',') {
        let arch = arch.trim();
        if arch.is_empty() {
            continue;
        }
        let arch_num: u32 = match arch.parse() {
            Ok(n) => n,
            Err(_) => {
                println!("cargo:warning=Skipping invalid CUDA_ARCHS entry: {}", arch);
                continue;
            }
        };
        if let Some(max) = max_compute
            && arch_num > max
        {
            println!(
                "cargo:warning=Skipping sm_{} (unsupported by detected CUDA toolkit, max: sm_{})",
                arch_num, max
            );
            continue;
        }
        flags.push(format!("-gencode=arch=compute_{},code=sm_{}", arch, arch));
    }

    // Generate forward-compatible PTX for each major architecture family that is
    // both present in the arch list and supported by the detected CUDA toolkit.
    let ptx_archs_env = env::var("CUDA_PTX_ARCHS").ok();
    let ptx_candidates: Vec<u32> = if let Some(ref ptx_env) = ptx_archs_env {
        ptx_env
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    } else {
        vec![90, 100, 120]
    };

    for &ptx_arch in &ptx_candidates {
        if let Some(max) = max_compute
            && ptx_arch > max
        {
            continue;
        }
        flags.push(format!(
            "-gencode=arch=compute_{},code=compute_{}",
            ptx_arch, ptx_arch
        ));
    }

    flags
}

fn add_cuda_library_paths() {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib", cuda_path);
    } else if let Ok(cuda_home) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
        println!("cargo:rustc-link-search=native={}/lib", cuda_home);
    } else {
        // Try standard paths
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
    }
}
