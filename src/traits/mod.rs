pub mod tensor;
pub mod algebra;

// Re-export so crate users don’t need to dig deep paths
pub use tensor::{HasTensorCore, TensorLike};
pub use algebra::{Ring, Field};
