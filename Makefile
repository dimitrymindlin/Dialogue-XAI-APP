# Makefile
.PHONY: build run

build:
	docker buildx build --platform linux/amd64 -t dialogue-llm-app --load .

remove:
	docker rm -f dialogue-llm-app || true

run: remove
	docker run -d --name dialogue-llm-app -e PORT=4001 -p 4001:4001 dialogue-llm-app

build-push:
	docker buildx build --platform linux/amd64 --no-cache -t dimidockmin/dimidockmin/dialogue-llm-app --push .