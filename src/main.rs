#![allow(dead_code)]
extern crate rulinalg;
use rulinalg::matrix::{Axes, Matrix, BaseMatrix, BaseMatrixMut};
use rulinalg::vector::Vector;
use rulinalg::io::csv::Reader;

extern crate rand;

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
use neural_network::NeuralNetwork;

/// A helper function for loading and parsing the CIFAR-10 dataset. The dataset
/// is comprised of five "batches" (binary files) of training data and one batch
/// of test data.
///
/// The first byte is the label of the first image, which is a number in the
/// range 0-9. The next 3072 bytes are the values of the pixels of the image.
/// The first 1024 bytes are the red channel values, the next 1024 the green,
/// and the final 1024 the blue. The values are stored in row-major order, so
/// the first 32 bytes are the red channel values of the first row of the image.
/*fn load_cifar(path_to_binaries: &str) -> std::io::Result<(Matrix<f64>, Vector<u8>)> {
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
}*/

fn main() {
    let rdr = Reader::from_file("../mnist/mnist_test_10.csv").unwrap().has_headers(false);
    let mnist = Matrix::<f64>::read_csv(rdr).unwrap();
    println!("MNIST dimensions: {} x {}", mnist.rows(), mnist.cols());
    let (labels_slice, data_slice) = mnist.split_at(1, Axes::Col);
    let mut xtr = data_slice.into_matrix();
    let labels = labels_slice.into_matrix();
    println!("\nTraining label dimensions: {} x {}\n{}", labels.rows(), labels.cols(), labels);

    // Normalize RGB data
    xtr = xtr.apply(&|x| { x / 255.0 * 0.99 });
    xtr = xtr.transpose();

    for (i, training_example) in xtr.col_iter().enumerate() {
        let mut ytr = Matrix::<f64>::zeros(labels.rows(), 1) + 0.01;
        let class_index = labels[[i as usize, 0]] as usize;
        ytr[[class_index, 0]] = 0.99;

        println!("\nTraining data #{}:\n{}", i, ytr);
    }

    let inputs = 4;
    let hidden = 3;
    let outputs = 3;
    let learning_rate = 0.3;
    let mut ann = NeuralNetwork::new(inputs, hidden, outputs, learning_rate);
    return;

    println!("\nStarting weights: inputs -> hidden");
    println!("{}", ann.weights_ih);

    println!("\nStarting weights: hidden -> outputs");
    println!("{}", ann.weights_ho);

    println!("------------------------------------------------");
    let xtr = Matrix::new(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let ytr = Matrix::new(3, 1, vec![1.0, 0.0, 0.0]);
    ann.train(&xtr, &ytr);

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
