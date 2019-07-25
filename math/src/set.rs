pub mod integer_set;
pub mod traits;

pub trait Set {
    fn is_empty(&self) -> bool;
}

