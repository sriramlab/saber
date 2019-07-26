pub mod trait_impl;

pub trait Constructable {
    fn new() -> Self;
}

pub trait Collecting<T> {
    fn collect(&mut self, item: T);
}

pub trait ToIterator<I: Iterator<Item=E>, E> {
    fn to_iter(&self) -> I;
}
