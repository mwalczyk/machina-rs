#![allow(dead_code)]
extern crate rulinalg;
use rulinalg::matrix::{Axes, Matrix, BaseMatrix};
use rulinalg::vector::Vector;

use std::error::Error;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::Path;

mod model;
mod nearest_neighbor;
mod neural_network;

use model::Supervised;
use nearest_neighbor::NearestNeighbor;

/// A helper function for loading and parsing the CIFAR-10 dataset. The dataset
/// is comprised of five "batches" (binary files) of training data and one batch
/// of test data.
///
/// The first byte is the label of the first image, which is a number in the
/// range 0-9. The next 3072 bytes are the values of the pixels of the image.
/// The first 1024 bytes are the red channel values, the next 1024 the green,
/// and the final 1024 the blue. The values are stored in row-major order, so
/// the first 32 bytes are the red channel values of the first row of the image.
fn load_cifar(path_to_binaries: &str) -> std::io::Result<(Matrix<f64>, Vector<u8>)> {
    let mut contents = Vec::new();

    for i in 1..6 {
        let path = format!("{}data_batch_{}.bin", path_to_binaries, i);
        let mut file = File::open(path)?;
        file.read_to_end(&mut contents)?;
    }

    // 10000 entries per binary, 5 binaries
    // Each entry is a 3073 element vector, where the first element is the label
    const NUMBER_OF_ENTRIES: usize = 50000;
    const LENGTH_OF_ENTRY: usize = 3073;

    let all_examples = Matrix::new(NUMBER_OF_ENTRIES, LENGTH_OF_ENTRY, contents);

    // Grab all of the columns (except the first) of the composite matrix and convert the
    // data into a floating point format
    let xtr = all_examples.split_at(1, Axes::Col).1.into_matrix().try_into::<f64>().unwrap();

    // Extract the first column of the composite matrix, which corresponds
    // to all of the labels of the training set
    let ytr = Vector::<u8>::from(all_examples.col(0));

    Ok((xtr, ytr))
}

fn main() {

    /*
    let (cifar_data, cifar_labels) = match load_cifar("../cifar-10/") {
        Err(why) => panic!("{:?}", why),
        Ok((xtr, ytr)) => (xtr, ytr)
    };

    println!("{} x {}", cifar_data.rows(), cifar_data.cols());

    // Create and train the classifier
    let mut knn = NearestNeighbor::new(3);
    knn.train(&xtr, &ytr);
    knn.predict(&xte);
    */

}
