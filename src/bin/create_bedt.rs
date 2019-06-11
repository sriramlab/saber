#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;

use clap::Arg;

use bio_file_reader::plink_bed::PlinkBed;
use saber::program_flow::OrExit;
use saber::util::{extract_str_arg, get_pheno_arr, get_bed_bim_fam_path};

fn main() {
    let app = clap_app!(create_bedt =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg bfile: --bfile <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
        (@arg out_path: --out <OUT> "required; output file path")
    );
    let matches = app.get_matches();

    let out_path = extract_str_arg(&matches, "out_path");
    let bfile = extract_str_arg(&matches, "bfile");
    let [bed_path, bim_path, fam_path] = get_bed_bim_fam_path(&bfile);
    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}\nout_path: {}", bed_path, bim_path, fam_path, out_path);
    let mut bed = PlinkBed::new(&bed_path, &bim_path, &fam_path)
        .unwrap_or_exit(None::<String>);
    bed.create_bed_t(&out_path).unwrap_or_exit(Some("failed to create bedt"));
}
