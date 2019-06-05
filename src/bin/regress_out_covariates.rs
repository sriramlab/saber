#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;
extern crate saber;

use std::fs::OpenOptions;
use std::io::{Write, BufWriter};

use ndarray_linalg::Solve;

use saber::program_flow::OrExit;
use saber::stats_util::{mean, std};
use saber::util::{extract_str_arg, get_pheno_arr, get_plink_covariate_arr};

fn main() {
    let matches = clap_app!(Saber =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg covariate_path: --covariate <BFILE> "required; covariate PLINK file path")
        (@arg pheno_filename: --pheno <PHENO> "required; each row is one individual containing one phenotype value")
        (@arg out_path: --out <PHENO> "required; output file path")
    ).get_matches();

    let pheno_filename = extract_str_arg(&matches, "pheno_filename");
    let out_path = extract_str_arg(&matches, "out_path");
    let covariate_path = extract_str_arg(&matches, "covariate_path");

    println!("pheno_filepath: {}\nout_path: {}\ncovariate_path: {}", pheno_filename, out_path, covariate_path);

    println!("\n=> generating the covariate array");
    let cov_arr = get_plink_covariate_arr(&covariate_path)
        .unwrap_or_exit(Some("faile to create the covariate matrix"));
    println!("covariate_arr.dim: {:?}", cov_arr.dim());

    println!("\n=> generating the phenotype array");
    let mut pheno_arr = get_pheno_arr(&pheno_filename)
        .unwrap_or_exit(Some("failed to get the phenotype array"));
    println!("pheno_arr.dim: {:?}", pheno_arr.dim());
    println!("\n=> normalizing the phenotypes");
    pheno_arr -= mean(pheno_arr.iter()) as f32;
    pheno_arr /= std(pheno_arr.iter(), 0) as f32;

    println!("\n=> calculating the residual phenotype array");
    let ay = cov_arr.t().dot(&pheno_arr);
    let projection_coefficient = (cov_arr.t().dot(&cov_arr)).solve_into(ay).unwrap();
    let projection = cov_arr.dot(&projection_coefficient);
    let residual = pheno_arr - projection;

    println!("\n=> writing the residual phenotypes to {}", out_path);
    let f = OpenOptions::new().truncate(true).create(true).write(true).open(out_path.as_str())
                              .unwrap_or_exit(Some(format!("failed to create file {}", out_path)));
    let mut buf = BufWriter::new(f);
    for val in residual.iter() {
        buf.write_fmt(format_args!("{}\n", val))
           .unwrap_or_exit(Some("failed to write to the output file"));
    }
}
