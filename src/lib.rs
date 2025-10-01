//! # Nimerica Tensor Core
//!
//! Proof-of-concept Rust tensor library with generic traits, views, and slicing.
//!
//! ## Features
//! - Generic over element type `T`
//! - Generic over rank `R`
//! - Owned tensors (`Tensor<T, R>`) + views
//! - Tracing-based logging
//! - Rich error handling (`TensorError`, `TensorResult<T>`)

pub mod utils;
pub mod traits;
pub mod tensor;

// Re-export common items for convenience
pub use utils::{logging, errors};
pub use utils::errors::{TensorError, TensorResult};
pub use utils::logging::{trace, debug, info, warn, error};

/// Initialize global logging system (safe to call multiple times).
pub fn init() {
    utils::logging::init_logging();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_logging_once() {
        // Even if called multiple times, should not panic
        init();
        init();
        trace!("Logging initialized in tests");
        assert!(true);
    }

    #[test]
    fn error_creation_message() {
        let err = TensorError::msg("custom failure");
        assert_eq!(format!("{}", err), "Tensor error: custom failure");
    }

    #[test]
    fn shape_mismatch_formatting() {
        let err = TensorError::ShapeMismatch { expected: 4, got: 6 };
        assert_eq!(format!("{}", err), "Shape mismatch: expected 4 elements, got 6");
    }
}
