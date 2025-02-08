.PHONY: build build-amd64 build-push run remove

# Build locally for Mac M1 (ARM64)
build:
	docker buildx build --platform linux/arm64 -t dialogue-llm-app --load .

# Build locally for x86 (optional)
build-amd64:
	docker buildx build --platform linux/amd64 -t dialogue-llm-app --load .

# Remove existing container
remove:
	docker rm -f dialogue-llm-app || true

# Run locally on Mac M1 (ARM64)
run: remove
	docker run -d --name dialogue-llm-app -e PORT=4001 -p 4001:4001 dialogue-llm-app

# Build and push a multi-arch image (ARM64 + AMD64) to Docker Hub
build-push:
	docker buildx build --platform linux/arm64,linux/amd64 --no-cache -t dimidockmin/dialogue-llm-app --push .