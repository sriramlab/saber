use std::ops::Deref;
use std::slice::Iter;

use crate::traits::{Collecting, Constructable};
use crate::traits::ToIterator;

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

impl<'a, 's: 'a, E> ToIterator<'s, Iter<'a, E>, &'a E> for Vec<E> {
    fn to_iter(&'s self) -> Iter<'a, E> {
        self.iter()
    }
}
