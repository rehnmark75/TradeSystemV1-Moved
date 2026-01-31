import { Pool } from "pg";

const connectionString =
  process.env.STRATEGY_CONFIG_DATABASE_URL ||
  "postgresql://postgres:postgres@postgres:5432/strategy_config";

export const strategyConfigPool = new Pool({
  connectionString,
  max: 10
});
