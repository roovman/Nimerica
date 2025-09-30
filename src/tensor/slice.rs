use crate::{Tensor, TensorView};
use crate::utils::logging::trace;
use crate::utils::logging::TensorError;

#[derive(Debug, Clone)]
pub enum IndexSpec {
    Index(usize),                 // single coordinate
    Range { start: usize, end: usize }, // [start, end)
    All,
}

impl<T, const R: usize> Tensor<T, R> {
    pub fn slice(&self, specs: [IndexSpec; R]) -> Result<TensorView<'_, T, R>, TensorError> {
        assert_eq!(specs.len(), R);

        let mut new_dims = self.dims;
        let new_strides = self.strides;
        let mut base_offset: usize = 0;

        for (i, spec) in specs.iter().enumerate() {
            match spec {
                IndexSpec::Index(idx) => {
                    if *idx >= self.dims[i] {
                        return Err(TensorError::OutOfBounds {
                            idx: vec![*idx],
                            dims: self.dims.to_vec(),
                        });
                    }
                    // Advance base offset
                    base_offset += idx * self.strides[i];
                    // Collapse this dimension
                    new_dims[i] = 1;
                }
                IndexSpec::Range { start, end } => {
                    if *end > self.dims[i] || *start >= *end {
                        return Err(TensorError::OutOfBounds {
                            idx: vec![*start, *end],
                            dims: self.dims.to_vec(),
                        });
                    }
                    // Advance base offset to `start`
                    base_offset += start * self.strides[i];
                    new_dims[i] = end - start;
                }
                IndexSpec::All => {
                    // Keep full dimension
                    new_dims[i] = self.dims[i];
                }
            }
        }

        trace!(
            "slice: specs={:?}, new_dims={:?}, strides={:?}, base_offset={}",
            specs,
            new_dims,
            new_strides,
            base_offset
        );

        // SAFETY: we validated indices, and offset is within bounds
        let base_ptr = unsafe { self.data.as_ptr().add(base_offset) };
        let total_len: usize = new_dims.iter().product();
        let slice = unsafe { std::slice::from_raw_parts(base_ptr, total_len) };

        Ok(TensorView::new(new_dims, new_strides, slice))
    }
}
