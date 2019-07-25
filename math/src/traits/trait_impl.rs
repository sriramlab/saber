use crate::traits::{Collecting, Constructable};

impl<T> Constructable for Vec<T> {
    type Output = Vec<T>;
    fn new() -> Self::Output {
        Vec::new()
    }
}

impl<T> Collecting<T> for Vec<T> {
    fn collect(&mut self, item: T) {
        self.push(item);
    }
}
