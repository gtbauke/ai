mod classification;
mod dataframes;
mod metrics;

use classification::knn::KNearestNeighbors;
use dataframes::DataFrame;
use metrics::accuracy_score;

fn main() {
    println!("Loading Fisher's Iris dataset from 'iris.csv'...\n");

    let df = match DataFrame::from_csv("iris.csv") {
        Ok(data) => data,
        Err(err) => {
            eprintln!("Error loading 'iris.csv': {err:?}");
            return;
        }
    };

    println!(
        "Dataset loaded successfully: {} rows, {} columns.",
        df.rows(),
        df.cols()
    );

    // Separate features and target label ("species")
    let features = match DataFrame::new(
        df.columns
            .iter()
            .filter(|c| c.name != "species")
            .cloned()
            .collect(),
    ) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("Error constructing features DataFrame: {err:?}");
            return;
        }
    };

    let label_series = match df.column("species") {
        Some(col) => col.clone(),
        None => {
            eprintln!("Error: column 'species' not found in dataset.");
            return;
        }
    };

    let labels = match DataFrame::new(vec![label_series]) {
        Ok(l) => l,
        Err(err) => {
            eprintln!("Error constructing labels DataFrame: {err:?}");
            return;
        }
    };

    let n_rows = df.rows();
    let k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, n_rows];
    println!("\nComparing KNN models for k = 1, 2, 3, 4, 5, 6, 7, 8, 9, {n_rows}:");
    println!("{:-<55}", "");
    println!("{:<6} | {:<12} | {:<20}", "k", "Accuracy", "Status");
    println!("{:-<55}", "");

    for &k in &k_values {
        let knn = match KNearestNeighbors::new(k) {
            Ok(knn_inst) => knn_inst,
            Err(err) => {
                eprintln!("Failed to initialize KNN for k={k}: {err:?}");
                continue;
            }
        };

        let model = match knn.fit(&features, &labels) {
            Ok(fitted) => fitted,
            Err(err) => {
                eprintln!("Failed to fit KNN model for k={k}: {err:?}");
                continue;
            }
        };

        let predictions = match model.predict(&features) {
            Ok(preds) => preds,
            Err(err) => {
                eprintln!("Failed to generate predictions for k={k}: {err:?}");
                continue;
            }
        };

        match accuracy_score(&labels, &predictions) {
            Ok(acc) => {
                let percentage = acc * 100.0;
                println!("{:<6} | {:>10.2}% | Evaluated successfully", k, percentage);
            }
            Err(err) => {
                eprintln!(
                    "{:<6} | {:>10}   | Error calculating accuracy: {err:?}",
                    k, "N/A"
                );
            }
        }
    }

    println!("{:-<55}", "");
}
