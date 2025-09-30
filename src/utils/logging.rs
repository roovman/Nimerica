use std::sync::Once;
use tracing_subscriber;

/// Initialize logging system.
///
/// Debug build → default = DEBUG.  
/// Release build → default = INFO.  
/// Can be overridden with RUST_LOG.
pub fn init_logging() {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        let default_level = if cfg!(debug_assertions) {
            "debug"
        } else {
            "info"
        };

        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
                    .add_directive(default_level.parse().unwrap()),
            )
            .init();
    });
}

pub use tracing::{info, debug, warn, error, trace};
