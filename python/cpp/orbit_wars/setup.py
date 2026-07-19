from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

_ROOT = Path(__file__).resolve().parent

setup(
    name="orbit_wars_cpp",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="orbit_wars_cpp",
            sources=[
                str(_ROOT / "bindings.cpp"),
                str(_ROOT / "io.cpp"),
                str(_ROOT / "library.cpp"),
                str(_ROOT / "simulation.cpp"),
                str(_ROOT / "masks.cpp"),
                str(_ROOT / "kaggle_integration.cpp"),
                str(_ROOT / "honest_shared_intercept.cpp"),
                str(_ROOT / "honest_shared_features.cpp"),
                str(_ROOT / "cpp_env_v2" / "cpp_env_static_cache_v2.cpp"),
                str(_ROOT / "cpp_env_v2" / "cpp_env_static_cache_v2_mask.cpp"),
                str(_ROOT / "cpp_env_v2" / "cpp_env_static_cache_v2_features.cpp"),
                str(_ROOT / "cpp_env_v2" / "cpp_env_live_v2.cpp"),
            ],
            extra_compile_args=[
                "-std=c++17",
                "-O3",
                "-g",
                "-fno-omit-frame-pointer",
                "-DNDEBUG",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
