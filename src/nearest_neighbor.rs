use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

use model::Supervised;

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
pub struct NearestNeighbor {
    pub k: u32,
    xtr: Option<Matrix<f64>>,
    ytr: Option<Vector<u32>>,
}

impl NearestNeighbor {
    pub fn new(k: u32) -> NearestNeighbor {
        NearestNeighbor { k: k, xtr: None, ytr: None }
    }
}

impl Default for NearestNeighbor {
    fn default() -> NearestNeighbor {
        NearestNeighbor { k: 1, xtr: None, ytr: None }
    }
}

impl Supervised<Matrix<f64>, Vector<u32>> for NearestNeighbor {
    fn train(&mut self, xtr: &Matrix<f64>, ytr: &Vector<u32>) {
        if xtr.rows() != ytr.size() {
            panic!("The number of training examples is not equal to
                    the number of training labels");
        }
        self.xtr = Some(xtr.clone());
        self.ytr = Some(ytr.clone());
    }

    fn predict(&self, xte: &Matrix<f64>) -> Vector<u32> {
        if let (Some(xtr), Some(ytr)) = (self.xtr.as_ref(), self.ytr.as_ref()) {
            if xtr.cols() != xte.cols() {
                panic!("The number of features in the training set is
                        not equal the number of features in the test set");
            }

            // Make sure that the output matches the dimensions of the input
            let mut ypred = Vector::<u32>::zeros(xte.rows());

            // Iterate through all of the test examples
            for (row_test_index, row_test) in xte.row_iter().enumerate() {

                let mut distances = Vector::<f64>::zeros(xtr.rows());

                // Then, iterate through all of the training examples and calculate
                // the L1 distance between the two data points
                for (row_train_index, row_train) in xtr.row_iter().enumerate() {
                    distances[row_train_index] = (row_test.sum() - row_train.sum()).abs();
                }

                // Find the index of the training example that was closest to this
                // test example
                let index_of_smallest = distances.argmin().0;

                // Find the training label that corresponds to that index and store it
                ypred[row_test_index] = ytr[index_of_smallest];

                println!("Test example at index {} will be assigned label {}", row_test_index, ypred[row_test_index]);
            }
            return ypred;
        }
        else {
            panic!("This classifier has not been trained");
        }
    }
}
