
use thiserror::Error;

/// Errors that can occur when constructing or manipulating tensors.
#[derive(Debug, Error)]
pub enum TensorError {
    /// Shape mismatch: e.g. creating tensor from data with wrong length,
    /// or trying to broadcast incompatible shapes.
    #[error("Shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    /// Out-of-bounds indexing: indices do not fit within tensor dims.
    #[error("Index out of bounds: attempted index {idx:?} in tensor with dims {dims:?}")]
    OutOfBounds { idx: Vec<usize>, dims: Vec<usize> },

    /// Strides / memory layout invalid or unsupported.
    #[error("Invalid or unsupported memory layout/strides")]
    InvalidLayout,

    /// Requested operation requires contiguity, but tensor is strided.
    #[error("Tensor is not contiguous in memory")]
    NonContiguous,

    /// Operation not supported for this backend / view type.
    #[error("Operation not supported for this tensor backend")]
    UnsupportedOperation,

    /// Arithmetic operation failed due to type or algebraic mismatch.
    #[error("Operation not defined for given element type")]
    InvalidOperation,

    /// Generic catch-all error with a human-readable message.
    #[error("Tensor error: {0}")]
    Message(String),
}

/// Convenient alias.
pub type TensorResult<T> = Result<T, TensorError>;

/// Helper to easily make a custom message error.
impl TensorError {
    pub fn msg<S: Into<String>>(s: S) -> Self {
        TensorError::Message(s.into())
    }
}
