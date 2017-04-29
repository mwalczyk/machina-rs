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
        if train_data.rows() != train_labels.size() {
            panic!("The number of training examples is not equal to
                    the number of training labels");
        }
        self.train_data = train_data.clone();
        self.train_labels = train_labels.clone();
    }

    fn predict(&self, test_data: &Matrix<f64>) -> Vector<u32> {
        let mut predictions = Vec::new();

        // Iterate through all of the test examples
        for row_test in test_data.row_iter() {

            // Then, iterate through all of the training examples
            for row_train in self.train_data.row_iter() {

                // Calculate the L1 distance between the two data points
                let distance = (row_test.sum() - row_train.sum()).abs();
                println!("distance: {:?}", distance);
            }
        }

        let pv = Vector::new(predictions);
        pv
    }
}


fn main() {
    let k = 1;
    let train_data = Matrix::new(3, 2, vec![1.0, 2.0,
                                            3.0, 4.0,
                                            5.0, 6.0]);
    let train_labels = Vector::new(vec![0, 0, 1]);

    // Create the classifier
    let knn = NearestNeighbor{ k, train_data, train_labels };

    let test_data = Matrix::new(2, 2, vec![1.0, 2.0,
                                           3.0, 4.0]);
    knn.predict(&test_data);
}
