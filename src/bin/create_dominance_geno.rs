use biofile::plink_bed::{PlinkBed, PlinkSnpType};
use clap::{clap_app, Arg};
use program_flow::{argparse::extract_str_arg, OrExit};

use saber::util::get_bed_bim_fam_path;

fn main() {
    let mut app = clap_app!(create_dominance_geno =>
        (version: "0.1")
        (author: "Aaron Zhou")
    );
    app = app
        .arg(
            Arg::with_name("plink_filename_prefix")
                .long("bfile")
                .short("b")
                .takes_value(true)
                .required(true)
                .help(
                    "If we have files named \n\
                    PATH/TO/x.bed PATH/TO/x.bim PATH/TO/x.fam \n\
                    then the <plink_filename_prefix> should be path/to/x",
                ),
        )
        .arg(
            Arg::with_name("out_path")
                .long("out")
                .short("o")
                .takes_value(true)
                .required(true)
                .help("output file path"),
        );
    let matches = app.get_matches();

    let out_path = extract_str_arg(&matches, "out_path");
    let bfile = extract_str_arg(&matches, "plink_filename_prefix");
    let (bed_path, bim_path, fam_path) = get_bed_bim_fam_path(&bfile);
    println!(
        "PLINK bed path: {}\n\
        PLINK bim path: {}\n\
        PLINK fam path: {}\n\
        out_path: {}",
        bed_path, bim_path, fam_path, out_path
    );
    let bed = PlinkBed::new(&vec![(
        bed_path.clone(),
        bim_path,
        fam_path,
        PlinkSnpType::Additive,
    )])
    .unwrap_or_exit(None::<String>);

    println!("\n=> writing the dominance genotype matrix to {}", out_path);
    bed.create_dominance_geno_bed(0, &out_path)
        .unwrap_or_exit(Some(format!(
            "failed to create the dominance genotype matrix for {}",
            bed_path
        )));
}
