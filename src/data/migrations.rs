use refinery::embed_migrations;
use rusqlite::Connection;

use crate::error::AppResult;

embed_migrations!("migrations");

pub fn run(connection: &mut Connection) -> AppResult<()> {
    migrations::runner().run(connection)?;
    Ok(())
}
