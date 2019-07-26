use std::slice::Iter;

use crate::sample::Sample;
use crate::set::traits::Finite;
use crate::traits::ToIterator;
use std::ops::Deref;

impl<'a, E: Clone> Sample<'a, Iter<'a, E>, &'a E, Vec<E>> for Vec<E> where &'a E: Deref {}

impl<T> Finite for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, 's: 'a, E> ToIterator<'s, Iter<'a, E>, &'a E> for Vec<E> {
    fn to_iter(&'s self) -> Iter<'a, E> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::sample::Sample;

    #[test]
    fn test_sample_vec() {
        let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sample_size = 3;
        let sample = v.sample_subset_without_replacement(sample_size).unwrap();
        println!("{:?}", sample);
        assert_eq!(sample.len(), sample_size);
        sample.iter().for_each(|x| assert!(v.contains(x)));
    }
}
