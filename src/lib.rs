#![feature(test)]
#[macro_use]
extern crate clap;
#[cfg(feature = "cuda")]
extern crate collenchyma as co;
#[cfg(feature = "cuda")]
extern crate collenchyma_blas as co_blas;
extern crate colored;
#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate num_traits;
extern crate rand;
extern crate time;

pub mod trace_estimator;
pub mod heritability_estimator;
pub mod mailman;
pub mod util;
pub mod program_flow;
pub mod simulation;
#[cfg(feature = "cuda")]
pub mod cublas;
