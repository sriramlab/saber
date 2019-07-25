pub mod trait_impl;

pub trait Constructable {
    type Output;
    fn new() -> Self::Output;
}

pub trait Collecting<T> {
    fn collect(&mut self, item: T);
}

pub trait ToIterator<I: Iterator<Item=E>, E> {
    fn iter(&self) -> I;
}
