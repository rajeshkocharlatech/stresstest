import streamlit as st
import textwrap
import json
import time

st.set_page_config(page_title="Accelerator Stress Test", layout="wide")

st.title("🚀 Accelerator Stress Test")
st.caption("Generate PyTorch-based accelerator stress workloads")

left, right = st.columns(2)

with left:
    st.subheader("🧩 Input Configuration")

    user_intent = st.selectbox(
        "What part of accelerator to stress?",
        [
            "Compute Tiles",
            "L2 SRAM",
            "Cluster NoC",
            "HBM (HBM2E PHY)",
            "CDMA",
            "CCP",
            "PCIe",
        ]
    )

    workload_blocks = st.multiselect(
        "Select workload blocks",
        [
            "GEMM",
            "Convolution",
            "Transformer",
            "All-Reduce",
            "Tensor Copy",
        ],
        default=["GEMM"]
    )

    reuse_heavy = st.checkbox("Reuse-heavy workloads")
    multi_stream = st.checkbox("Multi-stream tensor movement")
    large_tensor = st.checkbox("Large tensor read/write")
    explicit_memory = st.checkbox("Explicit memory / tensor staging")
    host_device = st.checkbox("Host ↔ Device transfers")

    generate = st.button("⚙️ Generate Stress Test")

with right:
    st.subheader("📤 Output")

    if generate:
        st.success("Stress workload generated")

        code_lines = [
            "import torch",
            "import time",
            "",
            'device = "cuda" if torch.cuda.is_available() else "cpu"',
            "",
            "def stress_test():",
            "    streams = []",
        ]

        num_streams = 4 if multi_stream else 1
        code_lines.append(f"    for i in range({num_streams}):")
        code_lines.append("        streams.append(torch.cuda.Stream())")
        code_lines.append("")
        code_lines.append("    for _ in range(10):")

        for i, block in enumerate(workload_blocks):
            stream_idx = i % num_streams if multi_stream else 0
            code_lines.append(f"        with torch.cuda.stream(streams[{stream_idx}]):")
            
            if block == "GEMM":
                size = 8192 if large_tensor else 4096
                code_lines.append(f"            a = torch.randn({size}, {size}, device=device)")
                code_lines.append(f"            b = torch.randn({size}, {size}, device=device)")
                code_lines.append("            c = torch.matmul(a, b)")
                if reuse_heavy:
                    code_lines.append("            d = torch.matmul(a, b)")

            elif block == "Convolution":
                batch, channels, h, w = (64, 3, 224, 224)
                if large_tensor:
                    batch = 128
                code_lines.append(f"            x = torch.randn({batch}, {channels}, {h}, {w}, device=device)")
                code_lines.append("            conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)")
                code_lines.append("            y = conv(x)")

            elif block == "Transformer":
                seq_len, dim = (512, 512)
                if large_tensor:
                    seq_len = 1024
                code_lines.append(f"            x = torch.randn({seq_len}, {dim}, device=device)")
                code_lines.append("            attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8).to(device)")
                code_lines.append("            y, _ = attn(x, x, x)")

            elif block == "All-Reduce":
                size = 4096 if not large_tensor else 8192
                code_lines.append(f"            t = torch.randn({size}, {size}, device=device)")
                if reuse_heavy:
                    code_lines.append("            t += t")

            elif block == "Tensor Copy":
                code_lines.append("            t1 = torch.randn(1024, 1024, device=device)")
                if host_device:
                    code_lines.append("            t2 = t1.to('cpu')")
                    code_lines.append("            t3 = t2.to('cuda')")

            if explicit_memory:
                code_lines.append("            t_mem = t1.clone() if 't1' in locals() else None")

        code_lines.append("        torch.cuda.synchronize()")
        code_lines.append("")
        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    start = time.time()")
        code_lines.append("    stress_test()")
        code_lines.append("    print('Execution time:', time.time() - start)")

        stress_code = "\n".join(code_lines)

        st.markdown("### 🧪 PyTorch Stress Test (Executable)")
        st.code(textwrap.dedent(stress_code), language="python")

        metrics = {
            "Target": user_intent,
            "Workloads": workload_blocks,
            "Reuse Heavy": reuse_heavy,
            "Multi Stream": multi_stream,
            "Large Tensor": large_tensor,
            "Explicit Memory": explicit_memory,
            "Host-Device Transfers": host_device,
            "Status": "Ready to Execute"
        }

        st.markdown("### 📊 Metrics (Sample)")
        st.json(metrics)

        metrics_json = json.dumps(metrics, indent=2)
        st.download_button(
            label="⬇️ Download Metrics (JSON)",
            data=metrics_json,
            file_name="stress_test_metrics.json",
            mime="application/json"
        )

    else:
        st.info("Configure inputs and click **Generate Stress Test**")
