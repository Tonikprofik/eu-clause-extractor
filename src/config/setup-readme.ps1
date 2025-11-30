# pshell
docker run -v "${PWD}\src\config\litellm_config.yaml:/app/config.yaml" -p 4000:4000 --rm ghcr.io/berriai/litellm:main-latest --config /app/config.yaml --detailed_debug