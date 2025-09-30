use std::sync::Once;
use tracing_subscriber::{EnvFilter, fmt};

pub fn init_logging() {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| {
                if cfg!(debug_assertions) {
                    EnvFilter::new("trace")   // default in debug
                } else {
                    EnvFilter::new("info")    // default in release
                }
            });

        fmt()
            .with_env_filter(filter)
            .init();
    });
}

/// Errors that can occur when constructing or using tensors.
#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch { expected: usize, got: usize },
    OutOfBounds { idx: Vec<usize>, dims: Vec<usize> },
    NonContiguous,
}


pub use tracing::{info, debug, warn, error, trace};
