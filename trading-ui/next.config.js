/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  basePath: "/trading",
  assetPrefix: "/trading",
  trailingSlash: true,
  output: "standalone"
};

module.exports = nextConfig;
