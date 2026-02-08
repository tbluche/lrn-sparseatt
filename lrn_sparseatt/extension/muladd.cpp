#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the STABLE_TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace extension_cpp {

torch::stable::Tensor sparse_matmul_cpu(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& indices) {


  STD_TORCH_CHECK(q.dim() == 2, "Expected q to be 2D");
  STD_TORCH_CHECK(k.dim() == 2, "Expected k to be 2D");
  STD_TORCH_CHECK(indices.dim() == 2, "Expected indices to be 2D");
  STD_TORCH_CHECK(q.sizes()[1] == k.sizes()[1], "Expected q and k to have the same number of columns");
  STD_TORCH_CHECK(q.scalar_type() == torch::headeronly::ScalarType::Float, "Expected q to be of type float");
  STD_TORCH_CHECK(k.scalar_type() == torch::headeronly::ScalarType::Float, "Expected k to be of type float");
  STD_TORCH_CHECK(indices.scalar_type() == torch::headeronly::ScalarType::Long, "Expected indices to be of type int");
  STD_TORCH_CHECK(q.device().type() == torch::headeronly::DeviceType::CPU, "Expected q to be on CPU");
  STD_TORCH_CHECK(k.device().type() == torch::headeronly::DeviceType::CPU, "Expected k to be on CPU");
  STD_TORCH_CHECK(indices.device().type() == torch::headeronly::DeviceType::CPU, "Expected indices to be on CPU");

  torch::stable::Tensor q_cont = torch::stable::contiguous(q);
  torch::stable::Tensor k_cont = torch::stable::contiguous(k);
  torch::stable::Tensor i_cont = torch::stable::contiguous(indices);

  // Indices is a 2D tensor of shape (num_outputs, 2), where each row is a pair of (q_index, k_index) indicating a non-zero entry in the output.
  int64_t num_outputs = indices.sizes()[0];
  torch::stable::Tensor result = torch::stable::empty(
    {num_outputs});
  //   dtype=q.scalar_type(), 
  //   device=q.device(),
  // );

  const float* q_ptr = q_cont.const_data_ptr<float>();
  const float* k_ptr = k_cont.const_data_ptr<float>();
  const int64_t* indices_ptr = i_cont.const_data_ptr<int64_t>();
  float* result_ptr = result.mutable_data_ptr<float>();
  

  int64_t q_dim = q.sizes()[1]; // number of columns in q and k
  float max = 0.0f;
  for (int64_t i = 0; i < num_outputs; i++) {
    int64_t q_index = indices_ptr[i * 2];
    int64_t k_index = indices_ptr[i * 2 + 1];
    float dot_product = 0.0f;
    for (int64_t j = 0; j < q_dim; j++) {
      dot_product += q_ptr[q_index * q_dim + j] * k_ptr[k_index * q_dim + j];
    }
    result_ptr[i] = dot_product;
    max = std::max(max, dot_product);
  }
  // Optional: apply softmax normalization to the result
  // for (int64_t i = 0; i < num_outputs; i++) {
  //   result_ptr[i] = std::exp(result_ptr[i] - max);
  // }
  return result;
}


// Defines the operators
STABLE_TORCH_LIBRARY(extension_cpp, m) {
  m.def("sparse_matmul(Tensor q, Tensor k, Tensor indices) -> Tensor");
}

// Registers CPU implementations for sparse_matmul
STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("sparse_matmul", TORCH_BOX(&sparse_matmul_cpu));
}

}