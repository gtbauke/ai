use crate::dataframes::errors::DataFrameError;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

pub mod errors;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ColumnData {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Utf8(Vec<String>),
    Bool(Vec<bool>),
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            Self::Int64(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Utf8(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Series {
    pub name: String,
    pub data: ColumnData,
}

impl Series {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn data(&self) -> &ColumnData {
        &self.data
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFrame {
    pub columns: Vec<Series>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl DataFrame {
    pub fn new(columns: Vec<Series>) -> Result<Self, DataFrameError> {
        if columns.is_empty() {
            return Ok(Self { columns });
        }

        let first_len = columns[0].data.len();
        for col in &columns[1..] {
            if col.data.len() != first_len {
                return Err(DataFrameError::InconsistentColumnLength);
            }
        }

        Ok(Self { columns })
    }

    #[inline]
    pub fn shape(&self) -> Shape {
        let cols = self.columns.len();
        let rows = self.columns.first().map_or(0, |col| col.data.len());

        Shape { rows, cols }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.shape().rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape().cols
    }

    pub fn column(&self, name: &str) -> Option<&Series> {
        self.columns.iter().find(|col| col.name == name)
    }

    pub fn from_series(series: Series) -> Result<Self, DataFrameError> {
        Self::new(vec![series])
    }

    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, DataFrameError> {
        let file = File::open(&path)
            .map_err(|_| DataFrameError::CsvNotFound(PathBuf::from(path.as_ref())))?;

        let mut reader = BufReader::new(file);
        let mut first_line = String::new();

        let bytes_read = reader
            .read_line(&mut first_line)
            .map_err(|_| DataFrameError::InvalidCsvFormat)?;

        if bytes_read == 0 {
            return Err(DataFrameError::MissingHeaders);
        }

        let headers = first_line
            .trim_end()
            .split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<String>>();

        if headers.is_empty() || (headers.len() == 1 && headers[0].is_empty()) {
            return Err(DataFrameError::MissingHeaders);
        }

        let num_cols = headers.len();
        let mut raw_rows: Vec<Vec<String>> = Vec::new();

        let mut line = String::new();
        while reader
            .read_line(&mut line)
            .map_err(|_| DataFrameError::InvalidCsvFormat)?
            > 0
        {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                line.clear();
                continue;
            }

            let row = trimmed
                .split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<String>>();

            line.clear();

            if row.len() != num_cols {
                return Err(DataFrameError::RowLengthDoesNotMatchHeadersLength);
            }

            raw_rows.push(row);
        }

        let mut columns = Vec::with_capacity(num_cols);
        for (col_idx, name) in headers.into_iter().enumerate() {
            let is_int = raw_rows
                .iter()
                .all(|row| row[col_idx].parse::<i64>().is_ok());
            let is_float = raw_rows
                .iter()
                .all(|row| row[col_idx].parse::<f64>().is_ok());
            let is_bool = raw_rows.iter().all(|row| {
                let val = row[col_idx].to_lowercase();
                val == "true" || val == "false"
            });

            let data = if is_int {
                let col_data = raw_rows
                    .iter()
                    .map(|row| row[col_idx].parse::<i64>().unwrap())
                    .collect::<Vec<i64>>();

                ColumnData::Int64(col_data)
            } else if is_float {
                let col_data = raw_rows
                    .iter()
                    .map(|row| row[col_idx].parse::<f64>().unwrap())
                    .collect::<Vec<f64>>();

                ColumnData::Float64(col_data)
            } else if is_bool {
                let col_data = raw_rows
                    .iter()
                    .map(|row| row[col_idx].to_lowercase() == "true")
                    .collect::<Vec<bool>>();

                ColumnData::Bool(col_data)
            } else {
                let col_data = raw_rows
                    .iter()
                    .map(|row| row[col_idx].clone())
                    .collect::<Vec<String>>();

                ColumnData::Utf8(col_data)
            };

            columns.push(Series { name, data });
        }

        DataFrame::new(columns)
    }

    // TODO: Use random shuffling to split the DataFrame into train and test sets
    pub fn split_test_train(&self, test_size: f64) -> Result<(Self, Self), DataFrameError> {
        if test_size < 0.0 || test_size > 1.0 {
            return Err(DataFrameError::InvalidTestSize);
        }

        let total_rows = self.rows();
        let test_rows = (total_rows as f64 * test_size).round() as usize;
        let train_rows = total_rows - test_rows;

        let mut train_columns = Vec::with_capacity(self.cols());
        let mut test_columns = Vec::with_capacity(self.cols());

        for col in &self.columns {
            let (train_data, test_data) = match &col.data {
                ColumnData::Int64(data) => {
                    let (train, test) = data.split_at(train_rows);
                    (
                        ColumnData::Int64(train.to_vec()),
                        ColumnData::Int64(test.to_vec()),
                    )
                }
                ColumnData::Float64(data) => {
                    let (train, test) = data.split_at(train_rows);
                    (
                        ColumnData::Float64(train.to_vec()),
                        ColumnData::Float64(test.to_vec()),
                    )
                }
                ColumnData::Utf8(data) => {
                    let (train, test) = data.split_at(train_rows);
                    (
                        ColumnData::Utf8(train.to_vec()),
                        ColumnData::Utf8(test.to_vec()),
                    )
                }
                ColumnData::Bool(data) => {
                    let (train, test) = data.split_at(train_rows);
                    (
                        ColumnData::Bool(train.to_vec()),
                        ColumnData::Bool(test.to_vec()),
                    )
                }
            };

            train_columns.push(Series {
                name: col.name.clone(),
                data: train_data,
            });

            test_columns.push(Series {
                name: col.name.clone(),
                data: test_data,
            });
        }

        Ok((
            Self {
                columns: train_columns,
            },
            Self {
                columns: test_columns,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_csv_iris() {
        let df = DataFrame::from_csv("iris.csv").expect("Should load iris.csv");
        assert_eq!(df.rows(), 150);
        assert_eq!(df.cols(), 5);
        assert!(df.column("sepal_length").is_some());
        assert!(df.column("species").is_some());
    }
}
