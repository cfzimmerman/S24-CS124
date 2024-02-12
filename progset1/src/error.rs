use thiserror::Error;

pub type PsetRes<T> = Result<T, PsetErr>;

/// Common error type for Progset 1 errors.
#[derive(Debug, Error)]
pub enum PsetErr {
    #[error("Static: {0}")]
    Static(&'static str),

    #[error("Context error: {0}")]
    Cxt(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),

    #[error(transparent)]
    Infallible(#[from] std::convert::Infallible),

    #[error(transparent)]
    Csv(#[from] csv::Error),
}
