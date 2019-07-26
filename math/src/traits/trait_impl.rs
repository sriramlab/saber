use crate::traits::{Collecting, Constructable};
use std::ops::Deref;

impl<T> Constructable for Vec<T> {
    fn new() -> Vec<T> {
        Vec::new()
    }
}

impl<T> Collecting<T> for Vec<T> {
    fn collect(&mut self, item: T) {
        self.push(item);
    }
}

impl<'a, T: Clone> Collecting<&'a T> for Vec<T> where &'a T: Deref {
    fn collect(&mut self, item: &'a T) {
        self.push((*item).clone());
    }
}
