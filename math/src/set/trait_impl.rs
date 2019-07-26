use crate::set::traits::Finite;

impl<T> Finite for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}
