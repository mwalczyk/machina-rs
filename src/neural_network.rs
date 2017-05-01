use rulinalg::matrix::{Matrix, BaseMatrix};
use rulinalg::vector::Vector;

use model::Supervised;

pub struct NeuralNetwork<'a> {
    /// The number of input neurons
    pub inputs: usize,

    /// A list that contains the number of neurons in each hidden layer, i.e. &[2, 3, 4]
    /// would create a neural network with 3 hidden layers that contain 2, 3, and 4 neurons,
    /// respectively
    pub hidden_layers: &'a [usize],

    /// The number of output neurons
    pub outputs: usize,

    /// A matrix that contains all of the weights
    weights: Matrix<f64>
}

impl<'a> Supervised<Matrix<f64>, Matrix<f64>> for NeuralNetwork<'a> {
    fn train(&mut self, xtr: &Matrix<f64>, ytr: &Matrix<f64>) {
        if xtr.cols() != self.inputs {
            panic!("The number of features in the provided training data
                    is not equal to the number of inputs to this network");
        }
        if ytr.cols() != self.outputs {
            panic!("The number of classes/outputs in the provided training labels
                    is not equal to the number of outputs to this network");
        }
        if xtr.rows() != ytr.rows() {
            panic!("The number of training examples does not equal the number
                    of training labels");
        }

        // Initialize weights

        // Perform N iterations (forward pass)

    }

    fn predict(&self, xte: &Matrix<f64>) -> Matrix<f64> {
        Matrix::new(2, 2, vec![1.0, 2.0,
                               3.0, 4.0])
    }
}
