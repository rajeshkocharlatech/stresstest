import streamlit as st
import textwrap

st.set_page_config(page_title="Accelerator Stress Test", layout="wide")

# ---------- HEADER ----------
st.title("🚀 Accelerator Stress Test")
st.caption("Generate PyTorch-based accelerator stress workloads")

# ---------- LAYOUT ----------
left, right = st.columns(2)

# ---------- INPUT PANEL ----------
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


# ---------- OUTPUT PANEL ----------
with right:
    st.subheader("📤 Output")

    if generate:
        st.success("Stress workload generated")

        stress_code = f"""
        import torch
        import time

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def stress_test():
            streams = []
            for i in range(4):
                streams.append(torch.cuda.Stream())

            for _ in range(10):
                with torch.cuda.stream(streams[0]):
                    a = torch.randn(4096, 4096, device=device)
                    b = torch.randn(4096, 4096, device=device)
                    c = torch.matmul(a, b)

                torch.cuda.synchronize()

        if __name__ == "__main__":
            start = time.time()
            stress_test()
            print("Execution time:", time.time() - start)
        """

        st.markdown("### 🧪 PyTorch Stress Test (Executable)")
        st.code(textwrap.dedent(stress_code), language="python")

        st.markdown("### 📊 Metrics (Sample)")
        st.json({
            "Target": user_intent,
            "Workloads": workload_blocks,
            "Reuse Heavy": reuse_heavy,
            "Multi Stream": multi_stream,
            "Large Tensor": large_tensor,
            "Explicit Memory": explicit_memory,
            "Host-Device Transfers": host_device,
            "Status": "Ready to Execute"
        })

    else:
        st.info("Configure inputs and click **Generate Stress Test**")
