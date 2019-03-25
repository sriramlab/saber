#[macro_use]
extern crate clap;

use bio_file_reader::plink_bed::PlinkBed;

fn main() {
    let matches = clap_app!(Saber =>
    (version: "0.1")
    (author: "Aaorn Zhou")
    (@arg plink_bed_filename: +required)
    ).get_matches();

    let plink_bed_filename = match matches.value_of("plink_bed_filename") {
        Some(plink_bed_filename) => plink_bed_filename.to_string(),
        None => {
            eprintln!("the argument plink_bed_filename is required");
            std::process::exit(1);
        }
    };

    println!("input filename: {}", plink_bed_filename);

    match PlinkBed::new(&plink_bed_filename) {
        Err(why) => { println!("{}", why); }
        Ok(_) => { println!("OK") }
    }
}
