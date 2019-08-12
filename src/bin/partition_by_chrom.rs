use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use clap::{Arg, clap_app};

use biofile::plink_bim::{CHROM_FIELD_INDEX, PlinkBim, VARIANT_ID_FIELD_INDEX};
use saber::program_flow::OrExit;
use saber::util::extract_str_arg;

fn main() {
    let mut app = clap_app!(partition_by_chrom =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg bim: --bim -b <BIM> "required; the PLINK bim file")
    );
    app = app.arg(
        Arg::with_name("out_path")
            .long("out")
            .short("o")
            .takes_value(true)
            .required(true)
            .help("output path; each line will have two fields: variant_id chrom_partition_assignment")
    );
    let matches = app.get_matches();

    let bim_path = extract_str_arg(&matches, "bim");
    let out_path = extract_str_arg(&matches, "out_path");

    println!("PLINK bim path: {}\nout_path: {}", bim_path, out_path);
    let mut bim = PlinkBim::new(&bim_path).unwrap_or_exit(Some("failed to create PlinkBim"));
    let mut writer = BufWriter::new(OpenOptions::new().write(true).create(true).truncate(true).open(&out_path)
                                                      .unwrap_or_exit(Some(format!("failed to create {}", bim_path))));
    assert_eq!(CHROM_FIELD_INDEX, 0);
    assert_eq!(VARIANT_ID_FIELD_INDEX, 1);
    for (i, line) in bim.lines().enumerate() {
        let l = line.unwrap_or_exit(Some("failed to get lines from the bim file object"));
        let mut toks = l.split_whitespace();

        let partition = toks.next()
                            .unwrap_or_exit(Some(format!("failed to extract the chrom from line {} in {}", i, bim_path)))
                            .to_string();

        let variant_id = toks.next()
                             .unwrap_or_exit(Some(format!("failed to extract variant id from line {} in {}",
                                                          i, bim_path)))
                             .to_string();

        writer.write_fmt(format_args!("{} {}\n", variant_id, partition))
              .unwrap_or_exit(Some(format!("failed to write to file: {}", out_path)));
    }
}
