#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;

use clap::Arg;

use bio_file_reader::plink_bed::PlinkBed;
use saber::program_flow::OrExit;
use saber::util::{extract_str_arg, get_bed_bim_fam_path, extract_optional_str_arg};

fn main() {
    let mut app = clap_app!(create_bedt =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg bfile: --bfile <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
        (@arg out_path: --out <OUT> "required; output file path")
    );
    app = app.arg(
        Arg::with_name("snp_chunk_size")
            .long("snp-chunk-size")
            .short("c")
            .takes_value(true)
            .help("larger values of `snp_byte_chunk_size` lead to faster performance, at the cost of higher memory requirement")
    );
    let matches = app.get_matches();

    let out_path = extract_str_arg(&matches, "out_path");
    let bfile = extract_str_arg(&matches, "bfile");
    let snp_chunk_size = match extract_optional_str_arg(&matches, "snp_chunk_size") {
        None => 4096,
        Some(s) => s.parse::<usize>().unwrap_or_exit(Some("failed to parse snp_chunk_size"))
    };
    let [bed_path, bim_path, fam_path] = get_bed_bim_fam_path(&bfile);
    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}\nout_path: {}\nsnp_chunk_size: {}",
             bed_path, bim_path, fam_path, out_path, snp_chunk_size);
    let mut bed = PlinkBed::new(&bed_path, &bim_path, &fam_path)
        .unwrap_or_exit(None::<String>);

    println!("\n=> writing the BED transpose to {}", out_path);
    bed.create_bed_t(&out_path, 4096).unwrap_or_exit(Some("failed to create bedt"));
}
