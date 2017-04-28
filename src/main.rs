extern crate rulinalg;

use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

trait Supervised<T, U> {
    fn train(&mut self, train_data: &T, train_labels: &U);
    fn predict(&self, test_data: &T) -> U;
}

trait Unsupervised {
    fn train(&mut self, train_data: &Matrix<f64>);
    fn predict(&self, test_data: &Vector<f64>) -> Vector<f64>;
}

/// A struct that implements the k-nearest neighbor algorithm.
///
/// The training phase for a k-nearest neighbor classifier simply
/// copies the training data and training labels into the struct.
///
/// The training data should be formatted as an MxN dimensional
/// matrix where each row is a training example and each column
/// is a feature. Similarly, the training labels should be a row
/// vector with M elements (one label per training example) that
/// are unsigned integers.
struct NearestNeighbor {
    k: u64,
    train_data: Matrix<f64>,
    train_labels: Vector<u32>,
}

impl Supervised<Matrix<f64>, Vector<u32>> for NearestNeighbor {
    fn train(&mut self, train_data: &Matrix<f64>, train_labels: &Vector<u32>) {
        self.train_data = train_data.clone();
        self.train_labels = train_labels.clone();
    }

    fn predict(&self, test_data: &Matrix<f64>) -> Vector<u32> {
        let mut predictions = Vec::new();

        for row in test_data.row_iter() {
            println!("{:?}", row);
        }

        let pv = Vector::new(predictions);
        pv
    }
}


fn main() {

    // A new matrix with 3 rows and 2 columns.
    let train_data = Matrix::new(6, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let train_labels = Vector::new(vec![0, 0, 0, 1, 1, 1]);

    let k = 1;


    println!("{}", k);
}
