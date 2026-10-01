use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricError {
    MismatchedLengths { len_true: usize, len_pred: usize },
    InvalidDataFrame,
}
