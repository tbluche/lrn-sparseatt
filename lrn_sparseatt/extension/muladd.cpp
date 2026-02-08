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
  torch::stable::Tensor result = torch::stable::empty({num_outputs});

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

torch::stable::Tensor sparse_matmul_vo_cpu(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& k_indices,
    const torch::stable::Tensor& q_offsets
  ) {
    STD_TORCH_CHECK(q.dim() == 2, "Expected q to be 2D");
    STD_TORCH_CHECK(k.dim() == 2, "Expected k to be 2D");
    STD_TORCH_CHECK(k_indices.dim() == 1, "Expected indices to be 1D");
    STD_TORCH_CHECK(q_offsets.dim() == 1, "Expected offsets to be 1D");

    STD_TORCH_CHECK(q.sizes()[1] == k.sizes()[1], "Expected q and k to have the same number of columns");
    STD_TORCH_CHECK(q.sizes()[0]== q_offsets.sizes()[0], "Expected q_offsets to have as many elements as the number of rows in q");

    STD_TORCH_CHECK(q.scalar_type() == torch::headeronly::ScalarType::Float, "Expected q to be of type float");
    STD_TORCH_CHECK(k.scalar_type() == torch::headeronly::ScalarType::Float, "Expected k to be of type float");
    STD_TORCH_CHECK(k_indices.scalar_type() == torch::headeronly::ScalarType::Long, "Expected indices to be of type int");
    STD_TORCH_CHECK(q_offsets.scalar_type() == torch::headeronly::ScalarType::Long, "Expected offsets to be of type int");

    STD_TORCH_CHECK(q.device().type() == torch::headeronly::DeviceType::CPU, "Expected q to be on CPU");
    STD_TORCH_CHECK(k.device().type() == torch::headeronly::DeviceType::CPU, "Expected k to be on CPU");
    STD_TORCH_CHECK(k_indices.device().type() == torch::headeronly::DeviceType::CPU, "Expected indices to be on CPU");
    STD_TORCH_CHECK(q_offsets.device().type() == torch::headeronly::DeviceType::CPU, "Expected offsets to be on CPU");

    torch::stable::Tensor q_cont = torch::stable::contiguous(q);
    torch::stable::Tensor k_cont = torch::stable::contiguous(k);

    // Indices is a 2D tensor of shape (num_outputs, 2), where each row is a pair of (q_index, k_index) indicating a non-zero entry in the output.
    int64_t num_outputs = k_indices.sizes()[0];
    int64_t num_q = q.sizes()[0];
    torch::stable::Tensor result = torch::stable::empty({num_outputs});

    const float* q_ptr = q_cont.const_data_ptr<float>();
    const float* k_ptr = k_cont.const_data_ptr<float>();
    const int64_t* indices_ptr = k_indices.const_data_ptr<int64_t>();
    const int64_t* offsets_ptr = q_offsets.const_data_ptr<int64_t>();
    float* result_ptr = result.mutable_data_ptr<float>();
    

    int64_t q_dim = q.sizes()[1]; // number of columns in q and k
    int64_t q_index = 0;
    int64_t current_offset = 0;

    float max = 0.0f;
    for (int64_t i = 0; i < num_outputs; i++) {
      // Move to the next q_index if we've passed the current offset
      if (q_index < num_q && i >= offsets_ptr[q_index]) {
        q_index++;
        current_offset = offsets_ptr[q_index];
      }
      int64_t k_index = indices_ptr[i];

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


torch::stable::Tensor sparse_attn_cpu(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& k_indices,
    const torch::stable::Tensor& q_offsets,
    double factor
  ) {
    STD_TORCH_CHECK(q.dim() == 2, "Expected q to be 2D");
    STD_TORCH_CHECK(k.dim() == 2, "Expected k to be 2D");
    STD_TORCH_CHECK(v.dim() == 2, "Expected v to be 2D");
    STD_TORCH_CHECK(k_indices.dim() == 1, "Expected indices to be 1D");
    STD_TORCH_CHECK(q_offsets.dim() == 1, "Expected offsets to be 1D");

    STD_TORCH_CHECK(q.sizes()[1] == k.sizes()[1], "Expected q and k to have the same number of columns");
    STD_TORCH_CHECK(k.sizes()[0] == v.sizes()[0], "Expected k and v to have the same number of rows");
    STD_TORCH_CHECK(q.sizes()[0]== q_offsets.sizes()[0], "Expected q_offsets to have as many elements as the number of rows in q");

    STD_TORCH_CHECK(q.scalar_type() == torch::headeronly::ScalarType::Float, "Expected q to be of type float");
    STD_TORCH_CHECK(k.scalar_type() == torch::headeronly::ScalarType::Float, "Expected k to be of type float");
    STD_TORCH_CHECK(v.scalar_type() == torch::headeronly::ScalarType::Float, "Expected v to be of type float");
    STD_TORCH_CHECK(k_indices.scalar_type() == torch::headeronly::ScalarType::Long, "Expected indices to be of type int");
    STD_TORCH_CHECK(q_offsets.scalar_type() == torch::headeronly::ScalarType::Long, "Expected offsets to be of type int");

    STD_TORCH_CHECK(q.device().type() == torch::headeronly::DeviceType::CPU, "Expected q to be on CPU");
    STD_TORCH_CHECK(k.device().type() == torch::headeronly::DeviceType::CPU, "Expected k to be on CPU");
    STD_TORCH_CHECK(v.device().type() == torch::headeronly::DeviceType::CPU, "Expected v to be on CPU");
    STD_TORCH_CHECK(k_indices.device().type() == torch::headeronly::DeviceType::CPU, "Expected indices to be on CPU");
    STD_TORCH_CHECK(q_offsets.device().type() == torch::headeronly::DeviceType::CPU, "Expected offsets to be on CPU");

    STD_TORCH_CHECK(factor > 0.0, "Expected factor to be positive");

    torch::stable::Tensor q_cont = torch::stable::contiguous(q);
    torch::stable::Tensor k_cont = torch::stable::contiguous(k);
    torch::stable::Tensor v_cont = torch::stable::contiguous(v);
    // Indices is a 2D tensor of shape (num_outputs, 2), where each row is a pair of (q_index, k_index) indicating a non-zero entry in the output.
    int64_t num_outputs = k_indices.sizes()[0];
    int64_t num_q = q.sizes()[0];
    torch::stable::Tensor result = torch::stable::empty({num_outputs});

    const float* q_ptr = q_cont.const_data_ptr<float>();
    const float* k_ptr = k_cont.const_data_ptr<float>();
    const float* v_ptr = v_cont.const_data_ptr<float>();
    const int64_t* indices_ptr = k_indices.const_data_ptr<int64_t>();
    const int64_t* offsets_ptr = q_offsets.const_data_ptr<int64_t>();
    float* result_ptr = result.mutable_data_ptr<float>();
    

    int64_t q_dim = q.sizes()[1]; // number of columns in q and k
    int64_t v_dim = v.sizes()[1]; // number of columns in v
    int64_t q_index = 0;
    int64_t current_offset = 0;

    double max = 0.0;
    for (int64_t i = 0; i < num_outputs; i++) {
      // Move to the next q_index if we've passed the current offset
      if (q_index < num_q && i >= offsets_ptr[q_index]) {
        q_index++;
        current_offset = offsets_ptr[q_index];
      }
      int64_t k_index = indices_ptr[i];

      float dot_product = 0.0f;
      for (int64_t j = 0; j < q_dim; j++) {
        dot_product += q_ptr[q_index * q_dim + j] * k_ptr[k_index * q_dim + j];
      }
      result_ptr[i] = dot_product / factor; // scale the dot product by the given factor (e.g., sqrt(d_model))
      max = std::max(max, static_cast<double>(dot_product) / factor);
    }
    
    // Compute the denominator for the softmax normalization
    torch::stable::Tensor denominator = torch::stable::empty({num_q});
    denominator = torch::stable::zero_(denominator);
    float* denominator_ptr = denominator.mutable_data_ptr<float>();

    q_index = 0;
    current_offset = 0;
    for (int64_t i = 0; i < num_outputs; i++) {
      double val = std::exp(static_cast<double>(result_ptr[i]) - max);
      result_ptr[i] = static_cast<float>(val);
      if (q_index < num_q && i >= offsets_ptr[q_index]) {
        q_index++;
        current_offset = offsets_ptr[q_index];
      }
      denominator_ptr[q_index] += val;
    }

    // Compute the final output by multiplying the normalized attention scores with the 
    // corresponding rows in v and summing them up for each query index.
    // TODO fix sizes for when there are different numbers of queries and keys/values
    torch::stable::Tensor output = torch::stable::new_zeros(v, v.sizes());
    float* output_ptr = output.mutable_data_ptr<float>();
    q_index = 0;
    current_offset = 0;
    for (int64_t i = 0; i < num_outputs; i++) {
      // Move to the next q_index if we've passed the current offset
      if (q_index < num_q && i >= offsets_ptr[q_index]) {
        q_index++;
        current_offset = offsets_ptr[q_index];
      }
      int64_t out_index = indices_ptr[i];

      float dot_product = 0.0f;
      for (int64_t j = 0; j < v_dim; j++) {
        output_ptr[q_index * v_dim + j] += result_ptr[i] * v_ptr[out_index * v_dim + j] / denominator_ptr[q_index];
      }
    }


    return output;

  }


// Defines the operators
STABLE_TORCH_LIBRARY(extension_cpp, m) {
  m.def("sparse_matmul(Tensor q, Tensor k, Tensor indices) -> Tensor");
  m.def("sparse_matmul_vo(Tensor q, Tensor k, Tensor k_indices, Tensor q_offsets) -> Tensor");
  m.def("sparse_attn(Tensor q, Tensor k, Tensor v, Tensor k_indices, Tensor q_offsets, float factor) -> Tensor");
}

// Registers CPU implementations for sparse_matmul
STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("sparse_matmul", TORCH_BOX(&sparse_matmul_cpu));
  m.impl("sparse_matmul_vo", TORCH_BOX(&sparse_matmul_vo_cpu));
  m.impl("sparse_attn", TORCH_BOX(&sparse_attn_cpu));
}

}