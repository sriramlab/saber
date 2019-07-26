pub mod ordered_integer_set;
pub mod traits;
pub mod trait_impl;

pub trait Set {
    fn is_empty(&self) -> bool;
}

