#![allow(dead_code)]
extern crate rulinalg;
use rulinalg::matrix::{Axes, Matrix, BaseMatrix, BaseMatrixMut};
use rulinalg::vector::Vector;
use rulinalg::io::csv::{Reader, Writer};

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

fn load_mnist(path: &str) -> (Matrix<f64>, Matrix<f64>) {
    let rdr = Reader::from_file(path).unwrap().has_headers(false);
    let mnist = Matrix::<f64>::read_csv(rdr).unwrap();

    println!("Loading MNIST data from: {}", path);
    println!("Dimensions: {} x {}", mnist.rows(), mnist.cols());

    // Set up matrices
    let (labels_slice, data_slice) = mnist.split_at(1, Axes::Col);
    let mut data = data_slice.into_matrix();
    let labels = labels_slice.into_matrix();

    // Normalize RGB data
    data = data.apply(&|x| { x / 255.0 * 0.99 });
    data = data.transpose();

    (data, labels)
}

fn main() {
    let (data, labels) = load_mnist("../mnist/mnist_train.csv");

    // Configure the neural network
    let inputs = 784;
    let hidden = 100;
    let outputs = 10;
    let learning_rate = 0.3;
    let mut ann = NeuralNetwork::new(inputs, hidden, outputs, learning_rate);

    println!("\nStarting training...");

    for (i, xtr) in data.col_iter().enumerate() {
        let mut ytr = Matrix::<f64>::zeros(outputs, 1) + 0.01;

        // A number in the range 0..9 corresponding to the class/label of
        // this training example
        let actual_class = labels[[i, 0]] as usize;

        ytr[[actual_class, 0]] = 0.99;

        ann.train(&(xtr.into_matrix()), &ytr);
    }

    let mut score_card: Vec<u32> = Vec::new();
    let (d_test, l_test) = load_mnist("../mnist/mnist_test.csv");
    for (i, xte) in d_test.col_iter().enumerate() {
        let actual_class = l_test[[i, 0]] as usize;
        let ypred = Vector::from( ann.predict(&xte.into_matrix()).col(0) );
        let predicted_class = ypred.argmax().0;

        // If the neural network predicted the correct class, append
        // a `1` to the list. Otherwise, append a `0`.
        if actual_class == predicted_class {
            score_card.push(1);
        }
        else {
            score_card.push(0);
        }
    }

    // Calculate the performance score
    let sum: u32 = score_card.iter().sum();
    let score: f64 = sum as f64 / score_card.len() as f64;
    println!("Final score: {}", score);

    // Save weight matrices
    let mut wtr = Writer::from_file("./weights_ih.csv").unwrap();
    ann.weights_ih.write_csv(&mut wtr).unwrap();

    wtr = Writer::from_file("./weights_ho.csv").unwrap();
    ann.weights_ho.write_csv(&mut wtr).unwrap();
}
