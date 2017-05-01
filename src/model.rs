/// All classifiers satisfy a common API.
///
/// In particular, they have a `train` function that takes the
/// data and labels to learn from. Internally, the class should
/// build some kind of model of the labels and how they can be
/// predicted from the data. The `predict` function takes new
/// data and predicts labels.
pub trait Supervised<T, U> {
    fn train(&mut self, xtr: &T, ytr: &U);
    fn predict(&self, xte: &T) -> U;
}

pub trait Unsupervised<T, U> {
    fn train(&mut self, xtr: &T);
    fn predict(&self, xte: &T) -> U;
}
