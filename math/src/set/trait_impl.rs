use std::collections::HashSet;

use crate::set::traits::{Finite, Set};

impl<T> Finite for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<T> Finite for HashSet<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<T> Set for HashSet<T> {
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::set::traits::{Cardinality, Countable, CountableType, Finite, HasCardinality, Set};

    #[test]
    fn test_finite_vec() {
        let v = vec![2, 5, 1, 7, -12, 3, 5, 71, -2];
        assert_eq!(Finite::size(&v), v.len());
        assert_eq!(Countable::count(&v), CountableType::Finite(v.len()));
        assert_eq!(Countable::is_finite(&v), true);
        assert_eq!(HasCardinality::get_cardinality(&v), Cardinality::Countable(CountableType::Finite(v.len())));
    }

    #[test]
    fn test_set() {
        let mut s = HashSet::new();
        assert_eq!(Set::is_empty(&s), true);
        s.insert(3);
        assert_eq!(Set::is_empty(&s), false);
    }
}
