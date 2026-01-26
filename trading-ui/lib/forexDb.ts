import { Pool } from "pg";

const connectionString =
  process.env.DATABASE_URL ||
  "postgresql://postgres:postgres@postgres:5432/forex";

export const forexPool = new Pool({
  connectionString,
  max: 10
});
