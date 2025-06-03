============
Installation
============

---------------------
Method 1. From source
---------------------

See `build from source <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/build.md>`_.

-------------------
Method 2. Using pip
-------------------

Prepare PyTorch container

.. code-block:: bash

    docker run --name triton-dist --ipc=host --network=host --privileged --cap-add=SYS_ADMIN --shm-size=10g --gpus=all -itd nvcr.io/nvidia/pytorch:25.04-py3 /bin/bash
    docker exec -it triton-dist /bin/bash

Then, please download and fix NVSHMEM manually (as we cannot do this for you due to NVSHMEM license requirements). See `prepare NVSHMEM <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/prepare_nvshmem.md>`_.

After that, install clang-19

.. code-block:: bash

    apt update
    apt install clang-19 llvm-19 libclang-19-dev

Then, pip install triton-dist.

.. code-block:: bash
    
    export NVSHMEM_SRC=/workspace/nvshmem
    export CC=clang-19
    export CXX=clang-19++
    pip install "git+https://github.com/ByteDance-Seed/Triton-distributed.git#subdirectory=python" --no-build-isolation --force-reinstall

