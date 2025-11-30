# Multi-stage Dockerfile for Next.js (standalone output)
# Build requires Node 18+ (using Node 20 Alpine)
FROM node:20-alpine AS deps
WORKDIR /app

# Install dependencies (lockfile optional)
COPY package.json ./
# If you add a lockfile later, also COPY it here and prefer `npm ci`
RUN npm install

# Copy the rest of the UI source and build in a separate stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Build Next.js app (next.config.mjs uses output='standalone')
RUN npm run build

# Final runtime image
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
# Copy Next.js standalone server and static assets
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
ENV PORT=3000
# Start the Next.js standalone server
CMD ["node", "server.js"]