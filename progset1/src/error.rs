use thiserror::Error;

pub type PsetRes<T> = Result<T, PsetErr>;

#[derive(Debug, Error)]
pub enum PsetErr {
    #[error("Static: {0}")]
    Static(&'static str),

    #[error("Context error: {0}")]
    Cxt(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}
