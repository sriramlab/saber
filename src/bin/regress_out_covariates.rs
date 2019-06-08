#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;
extern crate saber;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use ndarray_linalg::Solve;

use saber::util::matrix_util::normalize_vector_inplace;
use saber::program_flow::OrExit;
use saber::util::{extract_str_arg, get_plink_pheno_data, get_plink_covariate_arr};

fn main() {
    let matches = clap_app!(Saber =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg covariate_path: --covariate <BFILE> "required; covariate PLINK file path")
        (@arg pheno_path: --pheno <PHENO> "required; each row has three fields FID IID pheno")
        (@arg out_path: --out <PHENO> "required; output file path")
    ).get_matches();

    let pheno_path = extract_str_arg(&matches, "pheno_path");
    let covariate_path = extract_str_arg(&matches, "covariate_path");
    let out_path = extract_str_arg(&matches, "out_path");

    println!("phenotype filepath: {}\ncovariate filepath: {}\noutput filepath: {}",
             pheno_path, covariate_path, out_path);

    println!("\n=> generating the covariate array");
    let cov_arr = get_plink_covariate_arr(&covariate_path)
        .unwrap_or_exit(Some("faile to create the covariate matrix"));
    println!("covariate_arr.dim: {:?}", cov_arr.dim());

    println!("\n=> generating the phenotype array");
    let (header, fid_vec, iid_vec, mut pheno_arr) = get_plink_pheno_data(&pheno_path)
        .unwrap_or_exit(Some("failed to get the phenotype array"));
    println!("pheno_arr.dim: {:?}", pheno_arr.dim());
    println!("\n=> normalizing the phenotypes");
    normalize_vector_inplace(&mut pheno_arr, 0);

    println!("\n=> calculating the residual phenotype array");
    let ay = cov_arr.t().dot(&pheno_arr);
    let projection_coefficient = (cov_arr.t().dot(&cov_arr)).solve_into(ay).unwrap();
    let projection = cov_arr.dot(&projection_coefficient);
    let residual = pheno_arr - projection;

    println!("\n=> writing the residual phenotypes to {}", out_path);
    let f = OpenOptions::new().truncate(true).create(true).write(true).open(out_path.as_str())
                              .unwrap_or_exit(Some(format!("failed to create file {}", out_path)));
    let mut buf = BufWriter::new(f);
    buf.write_fmt(format_args!("{}\n", header))
       .unwrap_or_exit(Some("failed to write to the output file"));
    for (i, val) in residual.iter().enumerate() {
        buf.write_fmt(format_args!("{} {} {}\n", fid_vec[i], iid_vec[i], val))
           .unwrap_or_exit(Some("failed to write to the output file"));
    }
}
