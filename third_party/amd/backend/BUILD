# load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = ["@triton//:license"],
    # default_compatible_with = [# "@build-target"],
    # default_visibility = [
        # "//third_party/py/triton:__subpackages__",
        # "@triton//:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
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
)

cc_library(
    name = "headers",
    defines = [
        "__HIP_PLATFORM_AMD__",
    ],
    includes = [
        "include",
    ],
    textual_hdrs = glob(["include/**/*.h"]),
)

exports_files([
    "include/TDMCommon.h",
])
