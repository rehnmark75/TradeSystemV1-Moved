FROM nginx:latest

RUN apt-get update && \
    apt-get install -y certbot python3-certbot-nginx && \
    rm -rf /var/lib/apt/lists/*

# nginx.conf is bind-mounted from ./nginx/nginx.https.conf via docker-compose.
# Edits on the host take effect with: docker exec nginx nginx -s reload
RUN mkdir -p /var/www/certbot

EXPOSE 80 4443 4446 8080