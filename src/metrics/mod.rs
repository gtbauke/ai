use crate::{
    dataframes::{ColumnData, DataFrame},
    metrics::errors::MetricError,
};

pub mod errors;

pub fn accuracy_score(y_true: &DataFrame, y_pred: &DataFrame) -> Result<f64, MetricError> {
    if y_true.rows() != y_pred.rows() {
        return Err(MetricError::MismatchedLengths {
            len_true: y_true.rows(),
            len_pred: y_pred.rows(),
        });
    }

    if y_true.rows() == 0 {
        return Ok(1.0);
    }

    let true_col = y_true
        .columns
        .first()
        .ok_or(MetricError::InvalidDataFrame)?;
    let pred_col = y_pred
        .columns
        .first()
        .ok_or(MetricError::InvalidDataFrame)?;

    let correct_predictions = match (&true_col.data, &pred_col.data) {
        (ColumnData::Int64(t), ColumnData::Int64(p)) => {
            t.iter().zip(p.iter()).filter(|(a, b)| a == b).count()
        }
        (ColumnData::Utf8(t), ColumnData::Utf8(p)) => {
            t.iter().zip(p.iter()).filter(|(a, b)| a == b).count()
        }
        (ColumnData::Bool(t), ColumnData::Bool(p)) => {
            t.iter().zip(p.iter()).filter(|(a, b)| a == b).count()
        }
        (ColumnData::Float64(t), ColumnData::Float64(p)) => {
            t.iter().zip(p.iter()).filter(|(a, b)| a == b).count()
        }
        _ => return Err(MetricError::InvalidDataFrame),
    };

    Ok(correct_predictions as f64 / y_true.rows() as f64)
}

pub fn error_rate(y_true: &DataFrame, y_pred: &DataFrame) -> Result<f64, MetricError> {
    let accuracy = accuracy_score(y_true, y_pred)?;
    Ok(1.0 - accuracy)
}

pub fn confusion_matrix(y_true: &DataFrame, y_pred: &DataFrame) -> Result<DataFrame, MetricError> {
    if y_true.rows() != y_pred.rows() {
        return Err(MetricError::MismatchedLengths {
            len_true: y_true.rows(),
            len_pred: y_pred.rows(),
        });
    }

    todo!("Implement confusion matrix calculation");
}
