extern crate rulinalg;
use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

use std::error::Error;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::Path;

mod model;
mod nearest_neighbor;

use model::Supervised;
use nearest_neighbor::NearestNeighbor;

/// The first byte is the label of the first image, which is a number in the
/// range 0-9. The next 3072 bytes are the values of the pixels of the image.
/// The first 1024 bytes are the red channel values, the next 1024 the green,
/// and the final 1024 the blue. The values are stored in row-major order, so
/// the first 32 bytes are the red channel values of the first row of the image.
fn load_dataset(path: &str) -> std::io::Result<(Matrix<f64>, Vector<u8>)> {
    let mut file = File::open(path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;

    let all_examples = Matrix::new(10000, 3073, contents);

    let xtr = match all_examples.select_cols(&[1, 3072]).try_into::<f64>() {
        Ok(m) => m,
        Err(e) => panic!("Failed to convert training data into floating point format"),
    };
    let ytr = Vector::<u8>::from(all_examples.col(0));

    for i in 0..10000 {
        println!("{}: {}", ytr[i], xtr.row(i)[0]);
    }

    Ok((xtr, ytr))
}

fn main() {

    match load_dataset("../cifar-10/data_batch_1.bin") {
        Err(why) => panic!("{:?}", why),
        Ok(_) => println!("Loaded dataset"),
    }

    // Training data and labels
    let xtr = Matrix::new(3, 2, vec![1.0, 2.0,
                                            3.0, 4.0,
                                            5.0, 6.0]);
    let ytr = Vector::new(vec![0, 1, 2]);

    // Test data
    let xte = Matrix::new(2, 2, vec![1.0, 2.0,
                                           3.0, 4.0]);

    // Create and train the classifier
    let mut knn = NearestNeighbor::new(3);
    knn.train(&xtr, &ytr);
    knn.predict(&xte);
}
