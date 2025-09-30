pub mod data;
pub mod view;
pub mod slice;
pub mod layout;
pub mod index;
pub mod display;

// Re-export the main types so users don’t need to dive into submodules
pub use data::Tensor;
// pub use view::{TensorView, TensorViewMut};
// pub use slice::{SliceIx, ix};
// pub use layout::{Contiguous, Strided};
// pub use display::TensorDisplay;
