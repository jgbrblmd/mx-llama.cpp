# jgbrblmd/mx-llama.cpp:gfx906-rocm716
# llama.cpp (mx-llama.cpp, ROCm 7.16 build, gfx906) + minimal ROCm 7.16 runtime.
# Layout mirrors mxxm/mx-llama.cpp:gfx906, but the runtime libs are from
# /opt/rocm-10.1.0 (ROCm 7.16 SDK, "rocm 10.1.0" naming) kept at the same
# path so the binaries' RUNPATH resolves unchanged.

FROM ubuntu:24.04
ARG GIT_COMMIT=unknown
ARG GIT_BRANCH=master
ENV DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.title=mx-llama.cpp \
      org.opencontainers.image.description="Performance-tuned llama.cpp for AMD gfx906 (MI50/MI60), ROCm 7.16 runtime, multi-GPU TP" \
      org.opencontainers.image.revision=$GIT_COMMIT

# minimal runtime deps (same set as mxxm reference image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4t64 libelf1t64 libnuma1 libdrm2 libdrm-amdgpu1 \
        libzstd1 zlib1g libssl3 ca-certificates curl kmod \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ROCm 7.16 runtime (transitive ldd closure of the binaries, ~553MB
# libs + 36MB rocBLAS precompiled kernels for gfx906)
COPY rocm/ /opt/rocm-10.1.0/
RUN ln -s /opt/rocm-10.1.0 /opt/rocm

# llama.cpp: binaries + shared libs
COPY llama/bin/ /usr/local/bin/
COPY llama/lib/ /usr/local/lib/
RUN ldconfig

ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm-10.1.0/lib:/usr/local/lib \
    ROCM_PATH=/opt/rocm \
    HSA_OVERRIDE_GFX_VERSION=9.0.6 \
    GGML_ENABLE_CUSTOM_AR=1 \
    HSA_FORCE_FINE_GRAIN_PCIE=1 \
    GPU_MAX_HW_QUEUES=8 \
    LLAMA_ENABLE_MTP_OPT=1

CMD ["bash"]
