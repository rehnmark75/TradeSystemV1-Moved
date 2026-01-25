/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  basePath: "/stock-scanner",
  assetPrefix: "/stock-scanner",
  trailingSlash: true,
  output: "standalone"
};

module.exports = nextConfig;
