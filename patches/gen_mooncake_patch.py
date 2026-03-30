#!/usr/bin/env python3
"""Generate mooncake_rocm_rdma.patch by editing source in-place and diffing."""
import sys
import os
import shutil
import subprocess

MC_ROOT = "/sgl-workspace/Mooncake"
RDMA_CPP = os.path.join(MC_ROOT, "mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp")
CONFIG_CPP = os.path.join(MC_ROOT, "mooncake-transfer-engine/src/config.cpp")

def patch_rdma_context(path):
    with open(path) as f:
        lines = f.readlines()

    out = []
    replaced = False
    for line in lines:
        if not replaced and line.strip() == "#if !defined(WITH_NVIDIA_PEERMEM) && defined(USE_CUDA)":
            replaced = True
            out.append("#if defined(USE_HIP)\n")
            out.append("    // ROCm path: detect GPU vs CPU memory.\n")
            out.append("    // On AMD with Pensando ionic NICs, ibv_reg_mr(GPU VRAM) returns ENOMEM\n")
            out.append("    // because ionic lacks GPU Direct RDMA.\n")
            out.append("    hipPointerAttribute_t ptrAttrs;\n")
            out.append("    memset(&ptrAttrs, 0, sizeof(ptrAttrs));\n")
            out.append("    hipError_t hip_err = hipPointerGetAttributes(&ptrAttrs, addr);\n")
            out.append("\n")
            out.append("    if (hip_err == hipSuccess && ptrAttrs.type == hipMemoryTypeDevice) {\n")
            out.append("        mrMeta.addr = addr;\n")
            out.append("        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);\n")
            out.append("        if (!mrMeta.mr) {\n")
            out.append("            if (errno == ENOMEM) {\n")
            out.append("                LOG(WARNING)\n")
            out.append('                    << "ROCm: ibv_reg_mr(GPU) ENOMEM at " << addr\n')
            out.append('                    << " length=" << length\n')
            out.append('                    << " NIC does not support GPU Direct RDMA.";\n')
            out.append("            } else {\n")
            out.append('                PLOG(ERROR) << "ROCm: ibv_reg_mr(GPU) failed at " << addr;\n')
            out.append("            }\n")
            out.append("            return ERR_CONTEXT;\n")
            out.append("        }\n")
            out.append("    } else {\n")
            out.append("        mrMeta.addr = addr;\n")
            out.append("        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);\n")
            out.append("    }\n")
            out.append("#elif !defined(WITH_NVIDIA_PEERMEM) && defined(USE_CUDA)\n")
        else:
            out.append(line)

    with open(path, "w") as f:
        f.writelines(out)

def patch_config(path):
    with open(path) as f:
        lines = f.readlines()

    ionic_block = [
        "\n",
        "#if defined(USE_HIP)\n",
        "    // Auto-detect Pensando ionic NICs and reduce max_sge.\n",
        "    if (!max_sge_env) {\n",
        "        struct ibv_device **dev_list = ibv_get_device_list(nullptr);\n",
        "        if (dev_list) {\n",
        "            for (int i = 0; dev_list[i]; ++i) {\n",
        "                struct ibv_context *ctx = ibv_open_device(dev_list[i]);\n",
        "                if (!ctx) continue;\n",
        "                struct ibv_device_attr attr;\n",
        "                if (ibv_query_device(ctx, &attr) == 0) {\n",
        "                    // Pensando vendor_id = 0x1dd8\n",
        "                    if (attr.vendor_id == 0x1dd8 && config.max_sge > 2) {\n",
        "                        config.max_sge = 2;\n",
        '                        LOG(INFO) << "Detected Pensando ionic NIC, setting max_sge=2";\n',
        "                    }\n",
        "                }\n",
        "                ibv_close_device(ctx);\n",
        "            }\n",
        "            ibv_free_device_list(dev_list);\n",
        "        }\n",
        "    }\n",
        "#endif\n",
    ]

    out = []
    i = 0
    inserted = False
    while i < len(lines):
        out.append(lines[i])
        # Insert after the MC_MAX_SGE closing brace
        if (not inserted
            and "MC_MAX_SGE" in lines[i]
            and i + 6 < len(lines)):
            # Find the closing brace of the if block
            j = i + 1
            while j < len(lines) and lines[j].strip() != "}":
                out.append(lines[j])
                j += 1
            if j < len(lines):
                out.append(lines[j])  # closing }
                out.extend(ionic_block)
                inserted = True
                i = j + 1
                continue
        i += 1

    with open(path, "w") as f:
        f.writelines(out)

# Backup originals
shutil.copy2(RDMA_CPP, "/tmp/rdma_orig.cpp")
shutil.copy2(CONFIG_CPP, "/tmp/config_orig.cpp")

# Apply edits
patch_rdma_context(RDMA_CPP)
patch_config(CONFIG_CPP)

# Generate diffs
out_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/patches/mooncake_rocm_rdma.patch"

with open(out_path, "w") as f:
    r1 = subprocess.run(
        ["diff", "-u",
         "--label", "a/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp",
         "--label", "b/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp",
         "/tmp/rdma_orig.cpp", RDMA_CPP],
        capture_output=True, text=True
    )
    f.write(r1.stdout)

    r2 = subprocess.run(
        ["diff", "-u",
         "--label", "a/mooncake-transfer-engine/src/config.cpp",
         "--label", "b/mooncake-transfer-engine/src/config.cpp",
         "/tmp/config_orig.cpp", CONFIG_CPP],
        capture_output=True, text=True
    )
    f.write(r2.stdout)

# Restore originals
shutil.copy2("/tmp/rdma_orig.cpp", RDMA_CPP)
shutil.copy2("/tmp/config_orig.cpp", CONFIG_CPP)

print(f"Patch written to {out_path}")
print(f"rdma_context.cpp diff: {len(r1.stdout)} bytes")
print(f"config.cpp diff: {len(r2.stdout)} bytes")
