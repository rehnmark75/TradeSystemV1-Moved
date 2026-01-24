FROM node:20-alpine AS base

WORKDIR /app

COPY package.json tsconfig.json next.config.js ./
RUN npm install

COPY app ./app
COPY lib ./lib

RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "start"]
