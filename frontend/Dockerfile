# ── Build stage ─────────────────────────────────────────────────────────────
FROM node:20-alpine AS build

WORKDIR /app

# VITE_API_URL is baked into the bundle at build time (Vite inlines
# import.meta.env.* during `vite build`).  Override with:
#   docker build --build-arg VITE_API_URL=http://my-host:8000 .
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

# ── Serve stage ─────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
