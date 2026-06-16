// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 TRITON_INTEL_PREDICATED_STORE=1 triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=triton_gen.predicated_load --implicit-check-not=triton_gen.predicated_store

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: inactive_lane_early_return
  // CHECK: llvm.cond_br
  // CHECK: llvm.load
  // CHECK: llvm.store
  tt.func public @inactive_lane_early_return(%empty_row: i1, %ptrs: tensor<1024x!tt.ptr<f32>, #blocked>, %mask: tensor<1024xi1, #blocked>, %other: tensor<1024xf32, #blocked>) {
    cf.cond_br %empty_row, ^exit, ^live

  ^live:
    %val = tt.load %ptrs, %mask, %other : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %val, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    cf.br ^exit

  ^exit:
    tt.return
  }
}
