.PHONY: build build-amd64 build-push run remove

# Build locally for Mac M1 (ARM64)
build:
	docker buildx build --platform linux/arm64 \
		--build-arg BUILDPLATFORM=linux/arm64 \
		--build-arg TARGETPLATFORM=linux/arm64 \
		-t dialogue-llm-app \
		--no-cache --load .

# Build locally for x86 (optional)
build-amd64:
	docker buildx build --platform linux/amd64 \
		--build-arg BUILDPLATFORM=linux/amd64 \
		--build-arg TARGETPLATFORM=linux/amd64 \
		-t dialogue-llm-app \
		--no-cache --load .

# Remove existing container
remove:
	docker rm -f dialogue-xai-app || true

# Run locally
run: remove
	docker run -d --name dialogue-xai-app \
		-e PORT=4001 \
		-p 4001:4001 \
		--memory=4g \
		--cpus=2 \
		dialogue-xai-app

# Build and push multi-arch image
build-push:
	docker buildx build \
		--platform linux/arm64,linux/amd64 \
		--no-cache \
		-t dimidockmin/dialogue-xai-app \
		--push .