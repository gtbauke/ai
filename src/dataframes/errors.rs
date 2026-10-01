use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFrameError {
    InvalidCsvFormat,
    MissingHeaders,
    InconsistentColumnLength,
    RowLengthDoesNotMatchHeadersLength,
    InvalidLabels,
    InvalidTestSize,
    CsvNotFound(PathBuf),
}
