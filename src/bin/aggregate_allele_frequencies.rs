use analytic::histogram::Histogram;
use analytic::stats::mean;
use biofile::plink_bed::{PlinkBed, PlinkSnpType};
use clap::{Arg, clap_app};
use program_flow::argparse::{extract_numeric_arg, extract_str_arg};
use program_flow::OrExit;
use rayon::prelude::*;

use saber::util::get_bed_bim_fam_path;

fn main() {
    let mut app = clap_app!(aggregate_allele_frequencies =>
        (version: "0.1")
        (author: "Aaron Zhou")
    );
    app = app
        .arg(
            Arg::with_name("plink_filename_prefix")
                .long("bfile").short("b").takes_value(true).required(true)
                .help(
                    "If we have files named \n\
                    PATH/TO/x.bed PATH/TO/x.bim PATH/TO/x.fam \n\
                    then the <plink_filename_prefix> should be path/to/x"
                )
        )
        .arg(
            Arg::with_name("chunk_size")
                .long("chunk-size").takes_value(true).default_value("50")
        );
    let matches = app.get_matches();

    let bfile = extract_str_arg(&matches, "plink_filename_prefix");
    let chunk_size = extract_numeric_arg::<usize>(&matches, "chunk_size")
        .unwrap_or_exit(None::<String>);
    let (bed_path, bim_path, fam_path) = get_bed_bim_fam_path(&bfile);
    println!(
        "PLINK bed path: {}\n\
        PLINK bim path: {}\n\
        PLINK fam path: {}\n\
        chunk_size: {}",
        bed_path,
        bim_path,
        fam_path,
        chunk_size,
    );
    let bed = PlinkBed::new(&vec![(bed_path, bim_path, fam_path, PlinkSnpType::Additive)])
        .unwrap_or_exit(None::<String>);

    let frequencies: Vec<f64> = bed
        .col_chunk_iter(chunk_size, None)
        .into_par_iter()
        .flat_map(|snp_chunk| {
            snp_chunk.gencolumns()
                     .into_iter()
                     .map(|col| mean(col.iter()) / 2.)
                     .collect::<Vec<f64>>()
        })
        .collect();
    let histogram = Histogram::new(&frequencies, 20, 0., 1.).unwrap_or_exit(None::<String>);
    println!("minor allele frequency histogram:\n{}", histogram);
}
