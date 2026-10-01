use std::collections::{BinaryHeap, HashMap};

use crate::dataframes::{ColumnData, DataFrame, Series, errors::DataFrameError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KNearestNeighborsError {
    InvalidK,
    TrainingDataAndLabelsMismatch,
    FeatureDimensionMismatch { expected: usize, found: usize },
    EmptyData,
    NonNumericData,
    InvalidLabels,
    FeatureNotFound(String),
}

#[derive(Debug, Clone)]
pub struct KNearestNeighbors {
    k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNearestNeighborsModel {
    k: usize,
    training_data: DataFrame,
    labels: DataFrame,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct NeighborCandidate {
    index: usize,
    distance_sq: f64,
}

impl Eq for NeighborCandidate {}

impl PartialOrd for NeighborCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance_sq.partial_cmp(&other.distance_sq)
    }
}

impl Ord for NeighborCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance_sq
            .partial_cmp(&other.distance_sq)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl KNearestNeighbors {
    pub fn new(k: usize) -> Result<Self, KNearestNeighborsError> {
        if k == 0 {
            return Err(KNearestNeighborsError::InvalidK);
        }

        Ok(KNearestNeighbors { k })
    }

    pub fn fit(
        &self,
        training_data: &DataFrame,
        labels: &DataFrame,
    ) -> Result<KNearestNeighborsModel, KNearestNeighborsError> {
        if training_data.rows() != labels.rows() {
            return Err(KNearestNeighborsError::TrainingDataAndLabelsMismatch);
        }

        if training_data.rows() == 0 {
            return Err(KNearestNeighborsError::EmptyData);
        }

        if self.k > training_data.rows() {
            return Err(KNearestNeighborsError::InvalidK);
        }

        Ok(KNearestNeighborsModel {
            k: self.k,
            training_data: training_data.clone(),
            labels: labels.clone(),
        })
    }
}

impl KNearestNeighborsModel {
    pub fn predict(&self, test_data: &DataFrame) -> Result<DataFrame, KNearestNeighborsError> {
        let n_train = self.training_data.rows();
        let n_test = test_data.rows();
        let n_features = self.training_data.cols();

        if test_data.cols() != n_features {
            return Err(KNearestNeighborsError::FeatureDimensionMismatch {
                expected: n_features,
                found: test_data.cols(),
            });
        }

        if n_test == 0 {
            return Err(KNearestNeighborsError::EmptyData);
        }

        let mut train_cols: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for col in &self.training_data.columns {
            match &col.data {
                ColumnData::Int64(vals) => {
                    train_cols.push(vals.iter().map(|&v| v as f64).collect());
                }
                ColumnData::Float64(vals) => train_cols.push(vals.clone()),
                _ => return Err(KNearestNeighborsError::NonNumericData),
            }
        }

        let mut test_cols: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for train_col in &self.training_data.columns {
            let test_series = test_data
                .column(&train_col.name)
                .ok_or_else(|| KNearestNeighborsError::FeatureNotFound(train_col.name.clone()))?;

            match &test_series.data {
                ColumnData::Int64(vals) => {
                    test_cols.push(vals.iter().map(|&v| v as f64).collect());
                }
                ColumnData::Float64(vals) => test_cols.push(vals.clone()),
                _ => return Err(KNearestNeighborsError::NonNumericData),
            }
        }

        let label_series = self
            .labels
            .columns
            .first()
            .ok_or(KNearestNeighborsError::InvalidLabels)?;

        let result = match &label_series.data {
            ColumnData::Int64(train_labels) => {
                let mut predications = Vec::with_capacity(n_test);

                for test_idx in 0..n_test {
                    let mut dist_sq = vec![0.0; n_train];
                    for f in 0..n_features {
                        let q = test_cols[f][test_idx];
                        let train_feature = &train_cols[f];

                        for i in 0..n_train {
                            let diff = train_feature[i] - q;
                            dist_sq[i] += diff * diff;
                        }
                    }

                    let mut heap: BinaryHeap<NeighborCandidate> =
                        BinaryHeap::with_capacity(self.k + 1);
                    for (train_idx, &d2) in dist_sq.iter().enumerate() {
                        heap.push(NeighborCandidate {
                            index: train_idx,
                            distance_sq: d2,
                        });

                        if heap.len() > self.k {
                            heap.pop();
                        }
                    }

                    let mut vote_counts = HashMap::new();
                    while let Some(candidate) = heap.pop() {
                        let label = train_labels[candidate.index];
                        *vote_counts.entry(label).or_insert(0) += 1;
                    }

                    let winner = vote_counts
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(label, _)| label)
                        .ok_or(KNearestNeighborsError::EmptyData)?;

                    predications.push(winner);
                }

                DataFrame::new(vec![Series {
                    name: format!("predicted_{}", label_series.name),
                    data: ColumnData::Int64(predications),
                }])
            }
            ColumnData::Utf8(train_labels) => {
                let mut predictions = Vec::with_capacity(n_test);

                for test_idx in 0..n_test {
                    let mut dist_sq = vec![0.0; n_train];

                    for f in 0..n_features {
                        let q = test_cols[f][test_idx];
                        let train_feature = &train_cols[f];

                        for i in 0..n_train {
                            let diff = train_feature[i] - q;
                            dist_sq[i] += diff * diff;
                        }
                    }

                    let mut heap: BinaryHeap<NeighborCandidate> =
                        BinaryHeap::with_capacity(self.k + 1);

                    for (train_idx, &d2) in dist_sq.iter().enumerate() {
                        heap.push(NeighborCandidate {
                            index: train_idx,
                            distance_sq: d2,
                        });

                        if heap.len() > self.k {
                            heap.pop();
                        }
                    }

                    let mut vote_counts = HashMap::new();
                    while let Some(candidate) = heap.pop() {
                        let label = &train_labels[candidate.index];
                        *vote_counts.entry(label.clone()).or_insert(0) += 1;
                    }

                    let winner = vote_counts
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(label, _)| label)
                        .ok_or(KNearestNeighborsError::EmptyData)?;

                    predictions.push(winner);
                }

                DataFrame::new(vec![Series {
                    name: format!("predicted_{}", label_series.name),
                    data: ColumnData::Utf8(predictions),
                }])
            }
            _ => Err(DataFrameError::InvalidLabels),
        };

        match result {
            Ok(df) => Ok(df),
            Err(e) => match e {
                DataFrameError::InvalidLabels => Err(KNearestNeighborsError::InvalidLabels),
                _ => Err(KNearestNeighborsError::NonNumericData),
            },
        }
    }
}
