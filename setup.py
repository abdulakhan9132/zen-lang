"""Optional C extension builder for ZEN native math kernels.

Falls back cleanly to pure Python if no C compiler is installed on the host.
"""

from __future__ import annotations

import os
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build native math kernels when a C compiler exists; fall back silently otherwise."""

    def build_extensions(self) -> None:
        if os.environ.get("ZEN_SKIP_NATIVE_BUILD") == "1":
            return
        try:
            super().build_extensions()
        except Exception:
            return

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception:
            return


ext_modules = (
    [
        Extension(
            "zen.native._zen_kernels",
            sources=["zen/native/zen_kernels.c"],
            extra_compile_args=[] if os.name == "nt" else ["-O3"],
        )
    ]
    if os.environ.get("ZEN_BUILD_NATIVE") == "1"
    else []
)

setup(
    version="0.2.3",
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
)
