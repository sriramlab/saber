use std::fmt;

pub trait OrExit<T> {
    fn unwrap_or_exit<M: fmt::Display>(self, with_msg_prefix: Option<M>) -> T;
}

impl<T, E: fmt::Display> OrExit<T> for Result<T, E> {
    fn unwrap_or_exit<M: fmt::Display>(self, with_msg_prefix: Option<M>) -> T {
        match self {
            Err(why) => {
                match with_msg_prefix {
                    None => eprintln!("{}", why),
                    Some(msg) => eprintln!("{}: {}", msg, why)
                };
                std::process::exit(1);
            }
            Ok(value) => value
        }
    }
}

impl<T> OrExit<T> for Option<T> {
    fn unwrap_or_exit<M: fmt::Display>(self, with_msg_prefix: Option<M>) -> T {
        match self {
            None => {
                match with_msg_prefix {
                    None => eprintln!("expected the Option to have some value"),
                    Some(msg) => eprintln!("{}", msg)
                };
                std::process::exit(1);
            }
            Some(value) => value
        }
    }
}
