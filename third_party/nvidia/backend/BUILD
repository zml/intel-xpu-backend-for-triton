load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["@triton//:license"],
    default_visibility = [
        "//third_party/py/triton:__subpackages__",
    ],
)

pybind_extension(
    name = "cuda_utils",
    srcs = ["driver.c"],
    copts = select({
        "@triton//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
            "-Wno-missing-braces",
            "-Wno-sometimes-uninitialized",
        ],
    }),
    visibility = [
        "@jaxlib/triton:__subpackages__",
        "//third_party/py/triton:__subpackages__",
    ],
    deps = [
        # "@libcuda",
        "//third_party/absl/cleanup",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "files",
    srcs = glob(
        include = ["**/*.py"],
    ),
)

filegroup(
    name = "all-files",
    srcs = glob(
        include = ["**"],
    ),
    visibility = ["@triton//:__subpackages__"],
)
