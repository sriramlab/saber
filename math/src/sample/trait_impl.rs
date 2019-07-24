use crate::sample::Constructable;

impl<T> Constructable for Vec<T> {
    type Output = Vec<T>;
    fn new() -> Self::Output {
        Vec::new()
    }
}
