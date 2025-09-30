# Tensor Core (Proof of Concept)

This library implements a **minimal yet extensible tensor abstraction** in Rust.  
It’s designed as a proof of concept for a generic multi-dimensional array system with:  

---

### ✅ Core Features
- **Generic over element type `T`**  
  - Works with any data type (`i32`, `f64`, `bool`, custom structs, etc.).  
  - No numeric assumptions are made at the core level.  

- **Generic over rank `R` (compile-time constant)**  
  - Supports tensors of arbitrary dimensionality (scalars, vectors, matrices, N-D tensors).  
  - Shapes (`dims`) and strides (`strides`) stored as `[usize; R]`.  

- **Owned Tensors** (`Tensor<T, R>`)  
  - Backed by a contiguous `Vec<T>`.  
  - Automatic stride computation in row-major order.  
  - Error handling for shape mismatches.  

- **Tensor Views** (`TensorView` / `TensorViewMut`)  
  - Lightweight, non-owning slices into tensors.  
  - Immutable and mutable variants.  
  - Defined by raw pointer, dims, strides, and length.  
  - No assumptions on contiguity.  

- **Traits**  
  - `HasTensorCore<T>` — low-level access: `dims()`, `strides()`, `data()`, `data_mut()`.  
  - `TensorLike<T>` — high-level API: `rank()`, `len()`, `is_contiguous()`, `get()`, `set()`, etc.  
  - Blanket implementations allow all tensors and views to use the same API.  

- **Slicing API** (`slice()`)  
  - Index specifications:  
    - `IndexSpec::Index(usize)` – single coordinate.  
    - `IndexSpec::Range { start, end }` – sub-ranges.  
    - `IndexSpec::All` – full dimension.  
  - Produces new `TensorView` with adjusted dims/strides and base offset.  

- **Error Handling** (`TensorError`)  
  - `ShapeMismatch { expected, got }`  
  - `OutOfBounds { idx, dims }`  
  - `NonContiguous`  

- **Logging & Tracing**  
  - Integrated with [`tracing`](https://crates.io/crates/tracing).  
  - All operations emit `trace!`, `debug!`, and `error!` logs for step-by-step execution tracing.  
  - Enables *deterministic replay* of how shapes, strides, and indices are computed.  

---

### 🚀 Current Capabilities
- Allocate and initialize tensors (`Tensor::new`, `Tensor::defaulted`).  
- Read/write elements safely (`get`, `set`) or unsafely (`get_unchecked`, `set_unchecked`).  
- Create immutable and mutable views (`view`, `view_mut`).  
- Slice tensors into sub-views with arbitrary ranges and indices (`slice`).  
- Check memory layout contiguity (`is_contiguous`) and force materialization into contiguous storage (planned: `to_contiguous()`).  

---

### 🛠️ Design Principles
- **Minimal Core API**: all higher-level ops build on `HasTensorCore<T>`.  
- **Zero-cost Views**: slicing never copies data, just adjusts pointer, dims, and strides.  
- **Backend-Agnostic**: although currently implemented with `Vec<T>`, the API does not assume ownership. Future backends could include static arrays, GPU buffers, sparse maps, or custom allocators.  
- **Trace-Driven Debugging**: all indexing and stride math is observable through logs (`RUST_LOG=trace`).  

---

### 📦 Roadmap
- [ ] Implement `.to_contiguous()` to materialize strided views.  
- [ ] Add math traits (`Add`, `Mul`, `Dot`) for numeric tensors.  
- [ ] Explore alternative storage backends (sparse, column-major, GPU).  
- [ ] Python-like slicing syntax sugar (`t.slice([1, All, 2..5])`).  

---

This forms the **engineering backbone** for a future tensor algebra framework — simple enough to extend, but rigorous enough to debug every internal step with tracing.
