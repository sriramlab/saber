#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate saber;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use ndarray::Axis;
use ndarray_parallel::prelude::*;

use bio_file_reader::plink_bed::PlinkBed;
use saber::matrix_util::get_correlation;
use saber::program_flow::OrExit;
use saber::util::extract_str_arg;

fn main() {
    let matches = clap_app!(Saber =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg plink_filename_prefix: --bfile <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
        (@arg out_path: --out <OUT> "required; output path")
    ).get_matches();

    let out_path = extract_str_arg(&matches, "out_path");
    let plink_filename_prefix = extract_str_arg(&matches, "plink_filename_prefix");
    let plink_bed_path = format!("{}.bed", plink_filename_prefix);
    let plink_bim_path = format!("{}.bim", plink_filename_prefix);
    let plink_fam_path = format!("{}.fam", plink_filename_prefix);

    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}\nout_path: {}",
             plink_bed_path, plink_bim_path, plink_fam_path, out_path);

    let mut bed = PlinkBed::new(&plink_bed_path,
                                &plink_bim_path,
                                &plink_fam_path)
        .unwrap_or_exit(None::<String>);

    let geno_arr = bed.get_genotype_matrix()
                      .unwrap_or_exit(Some("failed to get the genotype matrix"));
    let (_num_people, num_snps) = geno_arr.dim();

    let f = OpenOptions::new().truncate(true).create(true).write(true).open(out_path.as_str())
                              .unwrap_or_exit(Some(format!("failed to create file {}", out_path)));
    let mut buf = BufWriter::new(f);

    for i in 0..num_snps - 1 {
        let snp_i = geno_arr.slice(s![.., i]);
        let rest = geno_arr.slice(s![.., i+1..]);

        let mut cor_vec = Vec::new();
        rest.axis_iter(Axis(1))
            .into_par_iter()
            .map(|col| get_correlation(&snp_i.to_owned(), &col.to_owned()))
            .collect_into_vec(&mut cor_vec);
        for val in cor_vec.into_iter() {
            buf.write_fmt(format_args!("{}\n", val))
               .unwrap_or_exit(Some("failed to write to the output file"));
        }
    }
}
