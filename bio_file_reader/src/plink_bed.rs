use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufReader, BufRead, Result};

pub struct PlinkBed {
    buf_reader: BufReader<File>
}

impl PlinkBed {
    pub fn new(filename: &String) -> std::io::Result<PlinkBed> {
        let buf_reader = BufReader::new(
            OpenOptions::new().read(true).open(filename.as_str())?
        );
        Ok(PlinkBed { buf_reader })
    }
}
