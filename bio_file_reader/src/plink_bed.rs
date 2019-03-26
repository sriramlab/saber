use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufReader, BufRead};
use std::io;
use std::fmt;
use std::convert::From;

pub enum Error {
    IO(io::Error),
    BadFormat(String),
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error::IO(error)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::IO(io_error) => write!(f, "IO error: {}", io_error),
            Error::BadFormat(why) => write!(f, "Bad format: {}", why)
        }
    }
}

pub struct PlinkBed {
    buf_reader: BufReader<File>
}

impl PlinkBed {
    pub fn new(filename: &String) -> Result<PlinkBed, Error> {
        let buf_reader = BufReader::new(
            OpenOptions::new().read(true).open(filename.as_str())?
        );
        Ok(PlinkBed { buf_reader })
    }
}
