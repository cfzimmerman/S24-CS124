use thiserror::Error;

pub type PsetRes<T> = Result<T, PsetErr>;

/// Common error type for Progset 1 errors.
#[derive(Debug, Error)]
pub enum PsetErr {
    #[error("Static: {0}")]
    Static(&'static str),

    #[error("Context error: {0}")]
    Cxt(String),

    #[error("Unrecognized CLI input")]
    InvalidInput,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    #[cfg(test)]
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}
