use std::sync::Once;
use tracing_subscriber::{fmt, EnvFilter};

/// Initialize tracing/logging system.
/// 
/// - Default log level: `trace` in debug, `info` in release.
/// - Override with `RUST_LOG` environment variable.
/// 
/// Safe to call multiple times (only initializes once).
pub fn init_logging() {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        // Choose default based on build profile
        let default_level = if cfg!(debug_assertions) {
            "debug"
        } else {
            "info"
        };

        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(default_level));

        fmt()
            .with_env_filter(filter)
            .with_target(true)       // show module path
            .with_level(true)        // show log level
            .compact()               // compact format for dev
            .init();
    });
}

// Re-export tracing macros so everything else can just `use crate::utils::logging::*;`
pub use tracing::{trace, debug, info, warn, error};
