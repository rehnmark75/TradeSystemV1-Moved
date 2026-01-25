FROM node:20-alpine AS base

WORKDIR /app

COPY package.json tsconfig.json next.config.js ./
RUN npm install

COPY app ./app
COPY lib ./lib
COPY components ./components

ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "start"]
