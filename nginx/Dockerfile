FROM nginx:latest

RUN apt-get update && \
    apt-get install -y certbot python3-certbot-nginx && \
    rm -rf /var/lib/apt/lists/*

COPY nginx.https.conf /etc/nginx/nginx.conf
RUN mkdir -p /var/www/certbot

EXPOSE 80 4443 4445