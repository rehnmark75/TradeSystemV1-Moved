import { Pool } from "pg";

const connectionString =
  process.env.STOCKS_DATABASE_URL ||
  "postgresql://postgres:postgres@postgres:5432/stocks";

export const pool = new Pool({
  connectionString,
  max: 10
});
